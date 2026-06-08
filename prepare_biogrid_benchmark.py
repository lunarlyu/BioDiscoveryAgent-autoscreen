"""Prepare per-screen BioGRID inputs for BioDiscoveryAgent.

BioDiscoveryAgent expects one dataset name at a time:
  datasets/task_prompts/<name>.json
  datasets/ground_truth_<name>.csv
  datasets/topmovers_<name>.npy

This script builds those files for the 348 AutoScreen BioGRID requests.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT.parent / "code"
BIOGRID_ROOT = CODE_ROOT / "benchmarks" / "BIOGRID"
SCREEN_TABLE_DIR = BIOGRID_ROOT / "BIOGRID-ORCS-ALL-homo_sapiens-1.1.16.screens"


def write_dataset(screen_id, request_text, top100_df, force=False):
    data_name = f"BIOGRID{screen_id}"
    prompt_path = ROOT / "datasets" / "task_prompts" / f"{data_name}.json"
    ground_truth_path = ROOT / "datasets" / f"ground_truth_{data_name}.csv"
    topmovers_path = ROOT / "datasets" / f"topmovers_{data_name}.npy"

    if (
        not force
        and prompt_path.exists()
        and ground_truth_path.exists()
        and topmovers_path.exists()
    ):
        return False

    screen_path = SCREEN_TABLE_DIR / f"BIOGRID-ORCS-SCREEN_{screen_id}-1.1.16.screen.tab.txt"
    if not screen_path.exists():
        raise FileNotFoundError(screen_path)

    screen_df = pd.read_csv(screen_path, sep="\t", dtype={"OFFICIAL_SYMBOL": str})
    screen_df = screen_df.dropna(subset=["OFFICIAL_SYMBOL"]).copy()
    screen_df["Score"] = pd.to_numeric(screen_df["SCORE.1"], errors="coerce").fillna(0.0)
    ground_truth = (
        screen_df[["OFFICIAL_SYMBOL", "Score"]]
        .rename(columns={"OFFICIAL_SYMBOL": "Gene"})
        .drop_duplicates(subset=["Gene"], keep="first")
    )

    hits = (
        top100_df.loc[top100_df["SCREEN_ID"].eq(screen_id), "OFFICIAL_SYMBOL"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .to_numpy()
    )

    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = {
        "Task": str(request_text),
        "Measurement": "the BioGRID ORCS phenotype score",
    }
    prompt_path.write_text(json.dumps(prompt, indent=2) + "\n")
    ground_truth.to_csv(ground_truth_path, index=False)
    np.save(topmovers_path, np.asarray(hits, dtype=str))
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    requests = pd.read_csv(BIOGRID_ROOT / "user_requests_activation_only.csv")
    top100 = pd.read_csv(BIOGRID_ROOT / "top_100_genes_per_screen.csv")

    written = 0
    for row in requests.itertuples(index=False):
        if write_dataset(int(row.ID), row.Request, top100, force=args.force):
            written += 1
    print(f"Prepared {len(requests)} BioGRID datasets ({written} written).")


if __name__ == "__main__":
    main()
