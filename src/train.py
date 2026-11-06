"""Estágio `train`: treina o modelo com os hiperparâmetros de params.yaml.

Saídas:
  models/model.pkl        -> pesos do modelo (binário, versionado pelo DVC)
  models/model_meta.json  -> linhagem: commit, hash dos dados, parâmetros (Git)
"""
import json
import platform
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from lineage import file_md5, git_commit, params_hash  # noqa: E402

TRAIN_DATA = "data/processed/train.csv"


def build_model(featurize: dict, train: dict):
    kind = train["model"]
    if kind == "random_forest":
        estimator = RandomForestClassifier(random_state=train["seed"], **train["random_forest"])
    elif kind == "logistic_regression":
        estimator = LogisticRegression(random_state=train["seed"], **train["logistic_regression"])
    else:
        raise ValueError(f"modelo desconhecido em params.yaml: {kind!r}")

    steps = [StandardScaler()] if featurize["scale"] else []
    return make_pipeline(*steps, estimator)


def main() -> None:
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    featurize, train = params["featurize"], params["train"]

    df = pd.read_csv(TRAIN_DATA)
    X, y = df.drop(columns="target"), df["target"]

    model = build_model(featurize, train)
    model.fit(X, y)

    out = Path("models")
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "model.pkl")

    used = {"featurize": featurize, "train": train}
    meta = {
        "model_type": train["model"],
        "hyperparameters": train[train["model"]],
        "params_hash": params_hash(used),
        "git_commit": git_commit(),
        "training_data": {"path": TRAIN_DATA, "md5": file_md5(TRAIN_DATA), "rows": len(df)},
        "features": list(X.columns),
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
        },
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                         encoding="utf-8", newline="\n")
    print(f"modelo {train['model']} treinado (params_hash={meta['params_hash']})")


if __name__ == "__main__":
    main()
