# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gke_ai_ml_gke_ray_raytrain_torchtpu_ray_train_llm_finetune]
import argparse
import os
import tempfile
import time

from datasets import load_dataset
import ray
import ray.train
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    FailureConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.torch import TorchConfig, TorchTrainer
import torch
from torch.distributed import ReduceOp, all_reduce, barrier
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader
import torch_tpu
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def build_alpaca_dataloader(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    max_samples: int,
    max_seq_len: int,
    batch_size_per_device: int,
) -> DataLoader:
    """Loads, formats, and tokenizes an instruction dataset for causal LM fine-tuning."""
    raw_dataset = load_dataset(dataset_name, split=f"train[:{max_samples}]")

    def format_and_tokenize(batch: dict) -> dict:
        prompts = [
            (
                f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
                if inp
                else f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            )
            for instruction, inp, output in zip(
                batch["instruction"], batch["input"], batch["output"]
            )
        ]
        return tokenizer(
            prompts,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )

    tokenized_dataset = raw_dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

    # Leave DataLoader num_workers=0 because Linux fork can deadlock after the
    # TPU runtime and Ray gRPC threads initialize.
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        drop_last=True,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
    )
    return ray.train.torch.prepare_data_loader(dataloader)


def shard_model_fsdp2(
    model: torch.nn.Module,
    world_size: int,
) -> None:
    """Shards a causal LM across a TPU slice using PyTorch FSDP2."""
    mesh = init_device_mesh("tpu", (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    text_model = model.model.language_model
    text_model.embed_tokens_per_layer.requires_grad_(False)

    for layer in text_model.layers:
        fully_shard(layer.mlp, mesh=mesh, mp_policy=mp_policy)
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        ignored_params=set(text_model.embed_tokens_per_layer.parameters()),
    )


def train_func(config: dict) -> None:
    """Per-worker distributed PyTorch training loop executed on each logical TPU device."""
    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    device = ray.train.torch.get_device()

    model_id = config["model_id"]
    dataset_name = config["dataset_name"]
    max_seq_len = config["max_seq_len"]
    batch_size_per_device = config["batch_size_per_device"]
    num_epochs = config["num_epochs"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataloader = build_alpaca_dataloader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        max_samples=config["max_samples"],
        max_seq_len=max_seq_len,
        batch_size_per_device=batch_size_per_device,
    )

    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        vision_config=None,
        audio_config=None,
    )
    model.config.get_text_config().use_cache = False

    start_epoch = 0
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            ckpt = torch.load(
                os.path.join(checkpoint_dir, "model.pt"),
                map_location="cpu",
            )
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            start_epoch = ckpt["epoch"]
        if world_rank == 0:
            print(
                f"[Rank 0/{world_size}] Resumed from checkpoint at epoch {start_epoch}"
            )

    model = model.to(device)
    shard_model_fsdp2(
        model=model,
        world_size=world_size,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    global_batch_size = batch_size_per_device * world_size
    if world_rank == 0:
        print(
            f"[Rank 0/{world_size}] Starting FSDP2 SFT for {model_id} on {dataset_name} "
            f"(global_batch_size={global_batch_size}, max_seq_len={max_seq_len}, "
            f"steps_per_epoch={len(dataloader)})"
        )

    for epoch in range(start_epoch, num_epochs):
        dataloader.sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.perf_counter()
        running_loss = torch.zeros(1, device=device, dtype=torch.float32)
        num_steps = 0

        for batch in dataloader:
            optimizer.zero_grad()
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.OVERRIDEABLE]
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                loss.backward()
            optimizer.step()
            running_loss += loss.detach().float()
            num_steps += 1

        all_reduce(running_loss, op=ReduceOp.AVG)
        avg_loss = running_loss.item() / num_steps
        epoch_time = time.perf_counter() - epoch_start
        tokens_per_sec = (
            num_steps * global_batch_size * max_seq_len
        ) / epoch_time

        metrics = {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "epoch_time_s": round(epoch_time, 2),
            "tokens_per_s": round(tokens_per_sec, 1),
            "steps_per_epoch": num_steps,
            "global_batch_size": global_batch_size,
        }
        if world_rank == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | loss={metrics['loss']:.4f} | "
                f"epoch_time={metrics['epoch_time_s']:.2f}s | "
                f"tokens/s={metrics['tokens_per_s']:.1f} | "
                f"steps={num_steps} | global_batch_size={global_batch_size}"
            )

        full_state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=True,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if world_rank == 0:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": full_state_dict,
                        "loss": metrics["loss"],
                    },
                    os.path.join(temp_checkpoint_dir, "model.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            del full_state_dict
            ray.train.report(metrics, checkpoint=checkpoint)
            barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Hugging Face CausalLM on TPUs with Ray Train and TorchTPU FSDP2."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-4-E2B-it",
        help="Hugging Face model ID to fine-tune.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="tatsu-lab/alpaca",
        help="Hugging Face instruction dataset name.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=os.environ.get("TPU_TOPOLOGY", "4x4"),
        help="TPU slice topology (for example, '4x4' on v6e or '2x2x2' on TPU7x).",
    )
    parser.add_argument(
        "--accelerator-type",
        type=str,
        default=os.environ.get("ACCELERATOR_TYPE", "TPU-V6E"),
        help="Ray TPU accelerator type ('TPU-V6E' or 'TPU-V7X').",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Total TPU worker processes across the slice.",
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size-per-device", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Maximum automatic controller restarts on worker failure.",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default=os.environ.get("STORAGE_PATH"),
        required="STORAGE_PATH" not in os.environ,
        help="Cloud Storage ('gs://...') or shared file system path for checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ray.init()
    run_name = f"torchtpu_gemma4_fsdp2_sft_{int(time.time())}"

    print(
        f"Starting Ray Train FSDP2 SFT job: model={args.model_id}, "
        f"dataset={args.dataset_name}, topology={args.topology}, "
        f"accelerator_type={args.accelerator_type}, "
        f"num_workers={args.num_workers}"
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "model_id": args.model_id,
            "dataset_name": args.dataset_name,
            "num_epochs": args.num_epochs,
            "batch_size_per_device": args.batch_size_per_device,
            "max_seq_len": args.max_seq_len,
            "max_samples": args.max_samples,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        torch_config=TorchConfig(backend="tpu_dist"),
        scaling_config=ScalingConfig(
            use_tpu=True,
            num_workers=args.num_workers,
            topology=args.topology,
            accelerator_type=args.accelerator_type,
            resources_per_worker={"TPU": 1},
        ),
        run_config=RunConfig(
            name=run_name,
            storage_path=args.storage_path,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="loss",
                checkpoint_score_order="min",
            ),
            failure_config=FailureConfig(max_failures=args.max_failures),
        ),
    )

    result = trainer.fit()
    print(f"Training completed! Final metrics: {result.metrics}")
    print(f"Saved checkpoint: {result.checkpoint}")


if __name__ == "__main__":
    main()
# [END gke_ai_ml_gke_ray_raytrain_torchtpu_ray_train_llm_finetune]
