from typing import Union, Dict, Any
import pathlib
import pickle
import time
from datetime import date
from pathlib import Path
import os

import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb

from prefect import flow, task
from prefect.deployments import Deployment
from prefect.artifacts import create_markdown_artifact
from prefect_email import EmailServerCredentials, email_send_message
from prefect.context import get_run_context
from prefect.server.schemas.schedules import CronSchedule
from prefect.deployments import run_deployment

CUR_PATH = Path(__file__).parent.resolve()
EMAIL_CREDENTIALS = os.getenv("EMAIL_CREDENTIALS", "email-credentials")
WORK_POOL_NAME = os.getenv("WORK_POOL_NAME", "mlops-zoomcamp")


### Q1 ###
@task(retries=3, retry_delay_seconds=2, name="Read taxi data")
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


### Q5, Q6
###
@flow(name="Exception email", log_prints=True)
def send_exception_email(exc):
    context = get_run_context()
    flow_run_name = context.flow_run.name
    email_server_credentials = EmailServerCredentials.load(EMAIL_CREDENTIALS)
    email_send_message(
        email_server_credentials=email_server_credentials,
        subject=f"Flow run {flow_run_name!r} failed",
        msg=f"Flow run {flow_run_name!r} failed due to {exc}.",
        email_to=email_server_credentials.username,
    )


### Q5, Q6
###
@flow(name="Succeed email", log_prints=True)
def send_ok_email():
    context = get_run_context()
    flow_run_name = context.flow_run.name
    email_server_credentials = EmailServerCredentials.load(EMAIL_CREDENTIALS)
    email_send_message(
        email_server_credentials=email_server_credentials,
        subject=f"Flow run {flow_run_name!r} succeeded",
        msg=f"Flow run {flow_run_name!r} succeeded.",
        email_to=email_server_credentials.username,
    )


@task
def add_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


@task(log_prints=True)
def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        markdown__rmse_report = f"""# RMSE Report

        ## Summary

        Duration Prediction 

        ## RMSE XGBoost Model

        | Region    | RMSE |
        |:----------|-------:|
        | {date.today()} | {rmse:.5f} |
        """
        ### Q4 ###
        create_markdown_artifact(
            key="duration-model-report", markdown=markdown__rmse_report
        )

    return rmse


@flow
def main_flow(
    train_path: Union[str, pathlib.PosixPath] = CUR_PATH
    / "data/green_tripdata_2023-01.parquet",
    val_path: Union[str, pathlib.PosixPath] = CUR_PATH
    / "data/green_tripdata_2023-02.parquet",
    send_email: bool = False,
) -> None:
    """The main training pipeline"""
    try:
        # MLflow settings
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("nyc-taxi-experiment")

        # Load
        df_train = read_data(str(train_path))
        df_val = read_data(str(val_path))

        # Transform
        X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

        # Train
        train_best_model(X_train, X_val, y_train, y_val, dv)
        ### Q5, Q6 - sending emails ###
        if send_email:
            send_ok_email()
    except Exception as exc:
        if send_email:
            send_exception_email(exc)


def apply_and_run_deployment(deployment_kwargs: Dict[str, Any]):
    deployment = Deployment(**deployment_kwargs)
    deployment.apply()
    deployment.to_yaml(f"{deployment.flow_name}-{deployment.name}.yaml")
    deployment_name = f"{deployment.flow_name}/{deployment.name}"
    response = run_deployment(deployment_name)
    print(response)


if __name__ == "__main__":
    """
    The code below is implemented for prefect-cloud
    To check all is work you need to: 1) login prefect cloud account, create workspace and set it locally
                                      2) create pool and start it locally `prefect agent start --pool $WORK_POOL_NAME`
                                        (pool name can be set via `export WORK_POOL_NAME=your_pool_name`)
                                      3) create e-mail credentials block for notifications (see `create_prefect_email_block.py`)
                                        (e-mail credentials can be set via `export EMAIL_CREDENTIALS=your_email_credentials_block_name`)
                                      4) run `python orchestration.py` in the new window
    """
    train_1_path = "data/green_tripdata_2023-01.parquet"
    train_2_path = "data/green_tripdata_2023-02.parquet"
    val_1_path = train_2_path
    val_2_path = "data/green_tripdata_2023-03.parquet"

    ### Q1 ###
    main_flow(train_path=train_1_path, val_path=val_1_path, send_email=False)
    # just to check flow
    main_flow(train_path=train_2_path, val_path=val_2_path, send_email=False)
    time.sleep(10)

    deployment_kwargs = dict(
        version=0,
        flow_name="main_flow",
        work_pool_name=WORK_POOL_NAME,
        path=str(CUR_PATH),
        entrypoint="orchestration.py:main_flow",
    )

    ### Q2, Q3 ###

    deployment_kwargs_q2_q3 = dict(
        name="Q2-Q3",
        parameters={
            "train_path": train_1_path,
            "val_path": val_1_path,
            "send_email": False,
        },
        schedule=CronSchedule(cron="0 9 3 * *"),
        description="Scheduled deployment",
    )
    deployment_kwargs_q2_q3.update(deployment_kwargs)
    apply_and_run_deployment(deployment_kwargs_q2_q3)

    time.sleep(10)

    ### Q4, Q5 ###

    deployment_kwargs_q4_q5 = dict(
        name="Q4-Q5",
        parameters={
            "train_path": train_2_path,
            "val_path": val_2_path,
            "send_email": True,
        },
        description="Notifications",
    )
    deployment_kwargs_q4_q5.update(deployment_kwargs)
    apply_and_run_deployment(deployment_kwargs_q4_q5)
