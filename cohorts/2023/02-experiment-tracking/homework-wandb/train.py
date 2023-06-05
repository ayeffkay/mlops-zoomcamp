import os
import pickle
import click


import wandb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from preprocess_data import DEFAULT_PARAMS


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option("--wandb_project", help="Name of Weights & Biases project")
@click.option("--wandb_entity", help="Name of Weights & Biases entity")
@click.option(
    "--data_artifact",
    help="Address of the Weights & Biases artifact holding the preprocessed data",
)
@click.option("--random_state", default=0, help="Random state")
@click.option("--max_depth", default=10, help="Max tree depth")
def run_train(
    wandb_project: str,
    wandb_entity: str,
    data_artifact: str,
    max_depth: int,
    random_state: int,
):
    # Initialize a Weights & Biases run
    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        job_type="train",
        config={"max_depth": max_depth, "random_state": random_state},
    ) as run:
        # Fetch the preprocessed dataset from artifacts
        artifact = run.use_artifact(data_artifact, type="preprocessed_dataset")
        data_path = artifact.download()

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Define the XGBoost Regressor Mode, train the model and perform prediction
        rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        mse = mean_squared_error(y_val, y_pred, squared=False)

        # DONE: Log `mse` to Weights & Biases under the key `"MSE"`
        run.log({"MSE": mse})

        model_name = DEFAULT_PARAMS["model_file"]
        with open(model_name, "wb") as f:
            pickle.dump(rf, f)

        # DONE: Log `regressor.pkl` as an artifact of type `model`
        artifact_name = DEFAULT_PARAMS["train_artifact_name"]
        run_artifact = wandb.Artifact(artifact_name, type="model")
        run_artifact.add_file(model_name)
        run.log_artifact(run_artifact)


if __name__ == "__main__":
    run_train()
