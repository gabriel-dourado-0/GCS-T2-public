"""Gera o dataset bruto em data/raw/dataset.csv.

Em um projeto real este arquivo viria de um data lake, de um bucket S3 etc.
Aqui usamos o dataset Breast Cancer Wisconsin (embutido no scikit-learn, sem
download) para que o exemplo rode offline.

A opção --extra-rows simula a chegada de uma NOVA VERSÃO dos dados (linhas
adicionais com ruído), para demonstrar o versionamento de datasets com DVC:

    python src/get_data.py                   # versão 1 dos dados
    python src/get_data.py --extra-rows 400  # versão 2 dos dados
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/raw/dataset.csv")
    args = parser.parse_args()

    df = load_breast_cancer(as_frame=True).frame

    if args.extra_rows:
        rng = np.random.default_rng(args.seed)
        extra = df.sample(n=args.extra_rows, replace=True, random_state=args.seed)
        features = extra.columns.drop("target")
        noise = rng.normal(0, 0.05, (len(extra), len(features))) * df[features].std().values
        extra[features] = extra[features] + noise
        df = pd.concat([df, extra], ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # float_format fixo => mesmo conteúdo gera sempre o mesmo hash
    df.to_csv(out, index=False, float_format="%.6f", lineterminator="\n")
    print(f"{out}: {len(df)} linhas, {df.shape[1]} colunas")


if __name__ == "__main__":
    main()
