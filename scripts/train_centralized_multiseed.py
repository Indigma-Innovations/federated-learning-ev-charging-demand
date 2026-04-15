#!/usr/bin/env python3
"""Run centralized training for multiple seeds and optionally multiple models."""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

DEFAULT_MODELS = [
    "linear_regression",
    "random_forest",
    "xgboost",
    "mlp",
    "gru",
    "transformer",
    "cnn1d"
]

NEURAL_MODELS = {"mlp", "transformer", "gru", "cnn1d"}


def parse_seeds(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one seed must be provided.")
    seeds = [int(part) for part in parts]
    return seeds


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run centralized training repeatedly across seeds. "
            "Model/data hyperparameters are forwarded to train_centralized.py."
        )
    )
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=(
            "Single model name, or 'all' to run: "
            "linear_regression, random_forest, xgboost, mlp, gru, transformer"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,7,42,101,123,173,561,999,2026",
        help="Comma-separated seeds. Default runs 10 seeds (0..9).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outputs/centralized",
        help="Same output root used by train_centralized.py.",
    )
    args, passthrough = parser.parse_known_args(argv)
    return args, passthrough


def resolve_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(DEFAULT_MODELS)
    return [model_arg]


def remove_checkpoint_if_present(outdir: Path, model: str, seed: int) -> None:
    pattern = f"best_{model}_*_{seed}_lambdaearly_*.pt"
    for path in outdir.glob(pattern):
        path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args, passthrough = parse_args(sys.argv[1:] if argv is None else argv)

    seeds = parse_seeds(args.seeds)
    models = resolve_models(args.model)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("Using deterministic data split from train_centralized.py (random_state=42).")
    print("This split is independent of --seed, so train/val/test do not change across runs.")

    for model in models:
        for seed in seeds:
            cmd = [
                sys.executable,
                "-m",
                "federated_acn.ml.train_centralized",
                "--data-path",
                args.data_path,
                "--model",
                model,
                "--seed",
                str(seed),
                "--outdir",
                str(outdir),
                *passthrough,
            ]
            print("\n>>> Running:", " ".join(cmd))
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                print(
                    f"Run failed for model={model}, seed={seed} with code={completed.returncode}",
                    file=sys.stderr,
                )
                return completed.returncode

            if model in NEURAL_MODELS:
                remove_checkpoint_if_present(outdir=outdir, model=model, seed=seed)

    print("\nFinished all requested centralized runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
