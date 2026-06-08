"""Run BioDiscoveryAgent zero-shot over the 348 AutoScreen BioGRID requests."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT.parent / "code"
BIOGRID_ROOT = CODE_ROOT / "benchmarks" / "BIOGRID"


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def convert_npy_to_autoscreen_csv(npy_path, output_csv):
    genes = [str(gene).strip() for gene in np.load(npy_path, allow_pickle=True)]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Gene": genes,
            "Number_of_Sources": 1,
            "Sources": "biodiscovery_agent_gpt54_zero_shot",
            "Weighted_Score": 1.0,
            "Expressed": "Yes",
            "Essential": pd.NA,
        }
    ).to_csv(output_csv, index=False)
    return genes


def load_metadata(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"metadata_error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--num-genes", type=int, default=1000)
    parser.add_argument("--prompt-tries", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--progress-dir",
        default="biodiscovery_gpt54_biogrid_progress",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CODE_ROOT / "baselines" / "results" / "results_batch_biodiscovery_gpt54_zero_shot"),
    )
    args = parser.parse_args()

    progress_dir = Path(args.progress_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_jsonl = progress_dir / "progress.jsonl"
    output_dir = Path(args.output_dir)

    requests = pd.read_csv(BIOGRID_ROOT / "user_requests_activation_only.csv")
    if args.limit is not None:
        requests = requests.iloc[args.start_index : args.start_index + args.limit]
    else:
        requests = requests.iloc[args.start_index :]

    env = os.environ.copy()
    env["BIODISCOVERY_LLM_PROVIDER"] = "openrouter"
    env.setdefault("OPENROUTER_MODEL", f"openai/{args.model}" if "/" not in args.model else args.model)

    total = len(requests)
    for ordinal, row in enumerate(requests.itertuples(index=False), start=1):
        screen_id = int(row.ID)
        data_name = f"BIOGRID{screen_id}"
        run_name = f"test_{data_name}"
        log_prefix = "biodiscovery_gpt54_biogrid"
        run_dir = ROOT / f"{log_prefix}_{data_name}" / run_name
        npy_path = run_dir / f"sampled_genes_{args.steps}.npy"
        metadata_path = run_dir / f"sampled_genes_{args.steps}_metadata.json"
        output_csv = output_dir / f"screen_{screen_id}" / "synthesized_genes.csv"

        if output_csv.exists() and not args.force:
            genes = pd.read_csv(output_csv)["Gene"].astype(str).tolist()
            record = {
                "event": "skip_existing",
                "screen_id": screen_id,
                "ordinal": ordinal,
                "total": total,
                "genes_saved": len(genes),
                "output_csv": str(output_csv),
                "run_dir": str(run_dir),
                "time": time.time(),
            }
            append_jsonl(progress_jsonl, record)
            print(json.dumps(record), flush=True)
            continue

        command = [
            sys.executable,
            "research_assistant.py",
            "--task",
            "perturb-genes-brief",
            "--model",
            args.model,
            "--run_name",
            run_name,
            "--data_name",
            data_name,
            "--steps",
            str(args.steps),
            "--num_genes",
            str(args.num_genes),
            "--log_dir",
            log_prefix,
            "--prompt_tries",
            str(args.prompt_tries),
            "--python",
            sys.executable,
        ]

        started = time.time()
        record = {
            "event": "start",
            "screen_id": screen_id,
            "ordinal": ordinal,
            "total": total,
            "command": command,
            "run_dir": str(run_dir),
            "time": started,
        }
        append_jsonl(progress_jsonl, record)
        print(json.dumps(record), flush=True)

        screen_log = progress_dir / f"screen_{screen_id}.log"
        with screen_log.open("a") as log_handle:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        elapsed = time.time() - started
        metadata = load_metadata(metadata_path)
        genes_saved = 0
        status = "missing_output"
        if npy_path.exists():
            genes = convert_npy_to_autoscreen_csv(npy_path, output_csv)
            genes_saved = len(genes)
            status = "converted"
        if proc.returncode != 0:
            status = "failed"

        record = {
            "event": "finish",
            "screen_id": screen_id,
            "ordinal": ordinal,
            "total": total,
            "returncode": proc.returncode,
            "status": status,
            "genes_saved": genes_saved,
            "elapsed_seconds": round(elapsed, 2),
            "metadata": metadata,
            "output_csv": str(output_csv),
            "screen_log": str(screen_log),
            "run_dir": str(run_dir),
            "time": time.time(),
        }
        append_jsonl(progress_jsonl, record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
