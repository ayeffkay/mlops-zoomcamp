#!/usr/bin/env python
# coding: utf-8
from typing import List
import os
import sys
import pathlib
import pickle
import pandas as pd

S3_NAME = os.getenv("S3_NAME", "nyc-duration")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
CUR_PATH = pathlib.Path(__file__).parent.resolve()


def get_input_path(year, month):
    default_input_pattern = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = f"s3://{S3_NAME}/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def prepare_data(df: pd.DataFrame, categorical: List[str]):
    df["duration"] = pd.to_datetime(df.tpep_dropoff_datetime) - pd.to_datetime(
        df.tpep_pickup_datetime
    )
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def read_data(filename: str, categorical: List[str]):
    if S3_ENDPOINT_URL:
        options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}
        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)
    df = prepare_data(df, categorical)
    return df


def save_data(df: pd.DataFrame, filename: str):
    if S3_ENDPOINT_URL:
        options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}
        df.to_parquet(
            filename,
            engine="pyarrow",
            compression=None,
            index=False,
            storage_options=options,
        )
    else:
        df.to_parquet(
            filename,
            engine="pyarrow",
            compression=None,
            index=False,
        )


def main(year: int, month: int):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    categorical = ["PULocationID", "DOLocationID"]

    with open(CUR_PATH / "model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_file, categorical)
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print("predicted mean duration:", y_pred.mean())
    print("predicted sum of durations: ", y_pred.sum())

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    save_data(df_result, output_file)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
