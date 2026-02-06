#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error

from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(
        project="nyc_airbnb",
        entity="c-ladebu32-western-governors-university",
        job_type="test_regression_model",
    )
    run.config.update(vars(args))

    logger.info("Downloading artifacts")
    model_local_path = run.use_artifact(args.mlflow_model).download()
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("price")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)

    y_pred = sk_pipe.predict(X_test)
    r_squared = sk_pipe.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)

    run.log({"r2": r_squared, "mae": mae})
    run.summary["r2"] = r_squared
    run.summary["mae"] = mae

    run.finish()


