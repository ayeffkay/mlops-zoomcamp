import os
import sys
import pandas as pd

S3_NAME = os.getenv("S3_NAME", "nyc-duration")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")


def save_df(input_file: str, year: int, month: int):
    df_input = pd.read_csv(input_file)
    options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}

    output_file = f"s3://{S3_NAME}/in/{year:04d}-{month:02d}.parquet"
    df_input.to_parquet(
        output_file,
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=options,
    )


if __name__ == "__main__":
    input_file = sys.argv[1]
    year = int(sys.argv[2])
    month = int(sys.argv[3])
    save_df(input_file, year, month)
