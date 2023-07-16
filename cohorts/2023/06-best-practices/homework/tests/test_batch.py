from typing import List, Tuple, Any
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = os.path.join(os.path.split(SCRIPT_DIR)[0])
sys.path.append(ROOT_DIR)

from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


@pytest.mark.parametrize(
    "data,columns",
    [
        [
            [
                (None, None, dt(1, 2), dt(1, 10)),
                (1, None, dt(1, 2), dt(1, 10)),
                (1, 2, dt(2, 2), dt(2, 3)),
                (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
                (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
                (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
            ],
            [
                "PULocationID",
                "DOLocationID",
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
            ],
        ],
    ],
)
def test_prepare_data(data: List[Tuple[Any]], columns: List[str]):
    df = pd.DataFrame(data, columns=columns)
    categorical = ["PULocationID", "DOLocationID"]
    prepared_df = prepare_data(df, categorical)

    expected_df = pd.DataFrame(
        data=[
            [
                "-1",
                "-1",
                pd.Timestamp("2022-01-01 01:02:00"),
                pd.Timestamp("2022-01-01 01:10:00"),
                8.0,
            ],
            [
                "1",
                "-1",
                pd.Timestamp("2022-01-01 01:02:00"),
                pd.Timestamp("2022-01-01 01:10:00"),
                8.0,
            ],
            [
                "1",
                "2",
                pd.Timestamp("2022-01-01 02:02:00"),
                pd.Timestamp("2022-01-01 02:03:00"),
                1.0,
            ],
        ],
        columns=[
            "PULocationID",
            "DOLocationID",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "duration",
        ],
    )
    Path(f"{ROOT_DIR}/output").mkdir(parents=True, exist_ok=True)
    expected_df.to_csv(f"{ROOT_DIR}/output/test_df.csv")

    assert (prepared_df == expected_df).values.all()
