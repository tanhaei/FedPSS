#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpss.config import load_config
from fedpss.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run federated patient similarity experiment.")
    parser.add_argument("--config", default="configs/test.yaml")
    parser.add_argument("--bioarc-jsonl", default=None)
    parser.add_argument("--outdir", default="outputs/test_run")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_experiment(cfg, outdir=args.outdir, bioarc_jsonl=args.bioarc_jsonl, device=args.device)
    print("Experiment complete")
    print(result["metrics"])

    # Generate standard PDF figures beside metrics.
    import subprocess

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_article_figures.py"), "--metrics", str(Path(args.outdir) / "metrics.json"), "--outdir", args.outdir],
        check=True,
    )


if __name__ == "__main__":
    main()
