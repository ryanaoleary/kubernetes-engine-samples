import ray
from ray import train, tune
from ray.tune import Tuner
from ray.train.xgboost import XGBoostTrainer

dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")

trainer = XGBoostTrainer(
    label_column="target",
    params={
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "max_depth": 4,
    },
    datasets={"train": dataset},
    scaling_config=train.ScalingConfig(num_workers=2),
)

# Create Tuner
tuner = Tuner(
    trainer,
    # Add some parameters to tune
    param_space={"params": {"max_depth": tune.choice([4, 5, 6])}},
    # Specify tuning behavior
    tune_config=tune.TuneConfig(metric="train-logloss", mode="min", num_samples=2),
)
# Run tuning job
tuner.fit()