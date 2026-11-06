"""Estágio `evaluate`: calcula métricas e aplica um quality gate.

As métricas ficam em metrics/metrics.json (versionado no Git), o que permite
comparar versões com `dvc metrics diff` e `dvc exp show`.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main() -> None:
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))["evaluate"]

    model = joblib.load("models/model.pkl")
    df = pd.read_csv("data/processed/test.csv")
    X, y = df.drop(columns="target"), df["target"]

    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }

    out = Path("metrics")
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    pd.DataFrame({"actual": y, "predicted": pred}).to_csv(
        out / "predictions.csv", index=False, lineterminator="\n"
    )
    print(json.dumps(metrics))

    if metrics["accuracy"] < params["min_accuracy"]:
        print(f"FALHA: accuracy {metrics['accuracy']} < mínimo {params['min_accuracy']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
