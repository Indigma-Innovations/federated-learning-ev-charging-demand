#!/usr/bin/env python3
"""Run federated simulation repeatedly across seeds and optionally models."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

DEFAULT_MODELS = [
    "mlp",
    "res-mlp",
    "dcn",
    "transformer",
    "cnn1d",
    "gru",
    "linear_regression",
    #"random_forest",
    "xgboost",
]


def parse_seeds(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one seed must be provided.")
    return [int(part) for part in parts]


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run federated simulation repeatedly across seeds. "
            "Additional CLI args are forwarded to run_simulation.py."
        )
    )
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=(
            "Single model name, or 'all' to run: "
            + ", ".join(DEFAULT_MODELS)
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,7,42,101,123,173,561,999,2026",
        help="Comma-separated seeds. Default runs 10 seeds.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--num-server-rounds", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
    parser.add_argument("--fraction-train", type=float, default=0.2)
    parser.add_argument("--fraction-evaluate", type=float, default=1.0)
    parser.add_argument(
        "--embedding",
        dest="use_embeddings",
        action="store_true",
        default=True,
        help="Enable categorical embeddings (default).",
    )
    parser.add_argument(
        "--no-embedding",
        dest="use_embeddings",
        action="store_false",
        help="Disable categorical embeddings.",
    )

    args, passthrough = parser.parse_known_args(argv)
    return args, passthrough


def resolve_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(DEFAULT_MODELS)
    return [model_arg]


def remove_checkpoint_if_present(model: str, seed: int) -> None:
    outdir = (Path.cwd() / "outputs" / "flower").resolve()
    if not outdir.exists():
        return

    patterns = (
        f"best_model_{model}_*_{seed}_*_lambdaearly_*.pt",
        f"best_model_{model}_*_{seed}_*_lambdaearly_*.json",
        f"best_model_{model}_*_{seed}_*_lambdaearly_*.pkl",
    )
    for pattern in patterns:
        for path in outdir.glob(pattern):
            path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args, passthrough = parse_args(sys.argv[1:] if argv is None else argv)

    seeds = parse_seeds(args.seeds)
    models = resolve_models(args.model)

    for model in models:
        for seed in seeds:
            cmd = [
                sys.executable,
                "-m",
                "federated_acn.fl.run_simulation",
                "--data-path",
                args.data_path,
                "--model-name",
                model,
                "--seed",
                str(seed),
                "--loss",
                args.loss,
                "--batch-size",
                str(args.batch_size),
                "--local-epochs",
                str(args.local_epochs),
                "--num-server-rounds",
                str(args.num_server_rounds),
                "--learning-rate",
                str(args.learning_rate),
                "--fraction-train",
                str(args.fraction_train),
                "--fraction-evaluate",
                str(args.fraction_evaluate),
            ]
            if args.use_embeddings:
                cmd.append("--embedding")
            else:
                cmd.append("--no-embedding")

            cmd.extend(passthrough)

            print("\n>>> Running:", " ".join(cmd))
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                print(
                    f"Run failed for model={model}, seed={seed} with code={completed.returncode}",
                    file=sys.stderr,
                )
                return completed.returncode
            remove_checkpoint_if_present(model=model, seed=seed)

    print("\nFinished all requested federated simulation runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
