#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import click
import logging
from logging import StreamHandler, Formatter

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
logger.addHandler(handler)


with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df

@click.command()
@click.option("--year", type=int, default=2022)
@click.option("--month", type=int, default=3)
@click.option("--taxi_type", type=click.Choice(['green', 'yellow'], case_sensitive=False), default='yellow')
@click.option("--file_to_save", type=str, default="")
def run_reader(year: int, month: int, taxi_type: str, file_to_save: str):
    file_name = "{trip_type}_tripdata_{year:04d}-{month:02d}.parquet".format(trip_type=taxi_type, year=year, month=month)
    df = read_data(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file_name}"
    )
    dicts = df[categorical].to_dict(orient="records")
    
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    logger.info(f"Mean predicted duration for {file_name} = {np.mean(y_pred):.4f} min.")
    
    ride_id = (f"{year:04d}/{month:02d}_" + df.index.astype("str")).to_numpy()
    data = np.concatenate((ride_id[:, np.newaxis], y_pred[:, np.newaxis]), axis=1)
    output_df = pd.DataFrame(data, columns=["ride_id", "predicted_duration"])
    
    if not len(file_to_save):
        file_to_save = file_name
    output_df.to_parquet(
        file_to_save, engine="pyarrow", compression=None, index=False
    )

if __name__ == '__main__':
    run_reader()