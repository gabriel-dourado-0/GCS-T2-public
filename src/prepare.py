"""Estágio `prepare`: divide o dataset bruto em treino e teste."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main() -> None:
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))["data"]

    df = pd.read_csv("data/raw/dataset.csv")
    train, test = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=df["target"],
    )

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False, lineterminator="\n")
    test.to_csv(out / "test.csv", index=False, lineterminator="\n")
    print(f"treino: {len(train)} linhas | teste: {len(test)} linhas")


if __name__ == "__main__":
    main()
