import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mp218.scripts.clean import CATEGORICAL, FEATURES, NUMERIC, TARGET, clean


def file_sha256(path: str) -> str:
    # hash of the ORIGINAL file bytes, before any cleaning
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


CANDIDATES = {
    "ridge_alpha_0.1": lambda seed: Ridge(alpha=0.1),
    "ridge_alpha_10": lambda seed: Ridge(alpha=10.0),
    "random_forest_depth_4": lambda seed: RandomForestRegressor(
        n_estimators=200, max_depth=4, random_state=seed),
}


def run(params_path: str = "params.yaml") -> None:
    with open(params_path) as f:
        p = yaml.safe_load(f)
    seed = p["model"]["seed"]
    raw_path = p["data"]["raw_train"]
    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment"])

    data_hash = file_sha256(raw_path)
    df = clean(pd.read_csv(raw_path))
    X_tr, X_va, y_tr, y_va = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=seed)

    results = []
    for name, make_model in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as active:
            pipe = build_pipeline(make_model(seed))
            pipe.fit(X_tr, y_tr)
            mae = mean_absolute_error(y_va, pipe.predict(X_va))
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_param("model_name", name)
            mlflow.log_metric("mae_valid", mae)
            mlflow.sklearn.log_model(
                pipe,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree"
                ]
            )
            results.append((mae, active.info.run_id, name))
            print(f"{name}: mae_valid={mae:.3f} run_id={active.info.run_id}")

    best_mae, best_run, best_name = min(results)
    model_name = p["mlflow"]["model_name"]
    mv = mlflow.register_model(f"runs:/{best_run}/model", model_name)
    MlflowClient().set_registered_model_alias(model_name, "champion", mv.version)

    best_pipe = mlflow.sklearn.load_model(f"runs:/{best_run}/model")
    out = Path(p["export"]["model_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, out)
    print(f"best={best_name} run_id={best_run} version={mv.version} mae={best_mae:.3f}")


if __name__ == "__main__":
    run()