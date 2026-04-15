import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from federated_acn.fl.task import build_client_id_columns, load_dataframe


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def compute_client_summary(
    df: pd.DataFrame, client_col: str, target_col: str
) -> pd.DataFrame:
    summary = (
        df.groupby(client_col)[target_col]
        .agg(["count", "mean", "min", "max", "std"])
        .reset_index()
        .rename(columns={client_col: "client_id", "count": "num_samples"})
        .sort_values("num_samples", ascending=False)
        .reset_index(drop=True)
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def _set_paper_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    plt.style.use("seaborn-v0_8-whitegrid")


def _paper_width_in() -> float:
    return 190 / 25.4

def _add_plot_client_labels(
    js_df: pd.DataFrame,
    sort_desc: bool = True,
) -> pd.DataFrame:
    plot_df = js_df.copy()
    if sort_desc:
        plot_df = plot_df.sort_values("js_divergence_vs_global", ascending=False).reset_index(drop=True)

    plot_df["plot_client_id"] = [f"Client{i+1}" for i in range(len(plot_df))]
    return plot_df

def plot_js_ranked_clients(
    js_df: pd.DataFrame,
    outpath: Path,
    title: str,
    js_threshold: float | None = None,
    top_k: int | None = None,
) -> None:
    _set_paper_plot_style()

    plot_df = _add_plot_client_labels(js_df, sort_desc=True)
    if top_k is not None:
        plot_df = plot_df.head(top_k).copy()

    width_in = _paper_width_in()
    height_in = 2.6
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    x = np.arange(len(plot_df))
    y = plot_df["js_divergence_vs_global"].to_numpy(dtype=float)

    ax.bar(
        x,
        y,
        color="#2E3A87",
        alpha=0.9,
        width=0.8,
    )

    if js_threshold is not None:
        ax.axhline(
            js_threshold,
            linestyle="--",
            linewidth=1.2,
            color="#C44E52",
            label=f"IID threshold = {js_threshold:.4f}",
        )
        ax.legend(frameon=True)

    ax.set_title(title, pad=4)
    #ax.set_xlabel("Client")
    ax.set_xlabel("")
    ax.set_ylabel("JS divergence")
    ax.set_xticks([])

    #ax.set_xticks(x)
    #ax.set_xticklabels(plot_df["plot_client_id"].tolist(), rotation=90)

    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.grid(False, axis="x")

    fig.tight_layout()
    outpath = outpath.with_suffix(".pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_js_vs_samples(
    js_df: pd.DataFrame,
    outpath: Path,
    title: str,
    js_threshold: float | None = None,
) -> None:
    _set_paper_plot_style()

    width_in = _paper_width_in()
    height_in = 2.6
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    x = js_df["num_samples"].to_numpy(dtype=float)
    y = js_df["js_divergence_vs_global"].to_numpy(dtype=float)

    ax.scatter(
        x,
        y,
        color="#2E3A87",
        alpha=0.8,
        s=18,
    )

    if js_threshold is not None:
        ax.axhline(
            js_threshold,
            linestyle="--",
            linewidth=1.2,
            color="#C44E52",
            label=f"IID threshold = {js_threshold:.4f}",
        )
        ax.legend(frameon=True)

    ax.set_title(title, pad=4)
    ax.set_xlabel("Number of samples per client")
    ax.set_ylabel("JS divergence")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    outpath = outpath.with_suffix(".pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary_distributions(summary_df: pd.DataFrame, outpath: Path, title: str) -> None:
    _set_paper_plot_style()

    cols = ["num_samples", "mean", "min", "max", "std"]
    width_in = _paper_width_in()
    height_in = 4.6

    fig, axes = plt.subplots(2, 3, figsize=(width_in, height_in), squeeze=False)
    axes_flat = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        ax.hist(
            summary_df[col].to_numpy(dtype=float),
            bins=30,
            color="#2E3A87",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_title(col.replace("_", " ").title(), pad=4)
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.grid(False, axis="x")

    axes_flat[-1].axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()

    outpath = outpath.with_suffix(".pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_top_clients_boxplot(
    df: pd.DataFrame,
    client_col: str,
    target_col: str,
    outpath: Path,
    top_k: int = 20,
) -> None:
    _set_paper_plot_style()

    top_clients = df[client_col].value_counts().head(top_k).index.astype(str).tolist()
    plot_df = df[df[client_col].astype(str).isin(top_clients)].copy()

    grouped_values = [
        plot_df.loc[
            plot_df[client_col].astype(str) == cid, target_col
        ].to_numpy(dtype=float)
        for cid in top_clients
    ]

    width_in = _paper_width_in()
    height_in = 2.8
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    bp = ax.boxplot(
        grouped_values,
        labels=top_clients,
        showfliers=False,
        patch_artist=True,
    )

    for box in bp["boxes"]:
        box.set(facecolor="#6C7BD9", alpha=0.5, edgecolor="#2E3A87", linewidth=0.8)
    for whisker in bp["whiskers"]:
        whisker.set(color="#2E3A87", linewidth=0.8)
    for cap in bp["caps"]:
        cap.set(color="#2E3A87", linewidth=0.8)
    for median in bp["medians"]:
        median.set(color="#C44E52", linewidth=1.0)

    ax.set_title(f"{target_col} distribution (top {top_k} clients by sample count)", pad=4)
    ax.set_xlabel("")
    ax.set_ylabel(target_col)
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.grid(False, axis="x")

    fig.tight_layout()
    outpath = outpath.with_suffix(".pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_js_distribution(
    js_df: pd.DataFrame,
    outpath: Path,
    title: str,
    js_threshold: float | None = None,
) -> None:
    _set_paper_plot_style()

    width_in = _paper_width_in()
    height_in = 2.6
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    ax.hist(
        js_df["js_divergence_vs_global"].to_numpy(dtype=float),
        bins=25,
        color="#2E3A87",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )

    if js_threshold is not None:
        ax.axvline(
            js_threshold,
            linestyle="--",
            linewidth=1.2,
            color="#C44E52",
            label=f"IID threshold = {js_threshold:.4f}",
        )
        ax.legend(frameon=True)

    ax.set_title(title, pad=4)
    ax.set_xlabel("JS divergence")
    ax.set_ylabel("Number of clients")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.grid(False, axis="x")

    fig.tight_layout()
    outpath = outpath.with_suffix(".pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _compute_histogram_js(
    values: np.ndarray,
    bin_edges: np.ndarray,
    global_hist: np.ndarray,
) -> float:
    client_hist, _ = np.histogram(values, bins=bin_edges, density=True)
    return _js_divergence(client_hist, global_hist)


def estimate_iid_js_baseline(
    df: pd.DataFrame,
    client_col: str,
    target_col: str,
    bins: int,
    n_simulations: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """
    Estimate a dataset-specific IID baseline for weighted JS divergence by
    randomly reassigning target values to clients while preserving client sizes.
    """
    valid_df = df[[client_col, target_col]].dropna().copy()
    client_sizes = valid_df.groupby(client_col).size().sort_index()
    eligible_clients = client_sizes[client_sizes >= 2]

    if len(eligible_clients) == 0:
        raise ValueError("No eligible clients with at least 2 samples for IID baseline estimation.")

    valid_df = valid_df[valid_df[client_col].isin(eligible_clients.index)].copy()
    global_values = valid_df[target_col].to_numpy(dtype=float)

    bin_edges = np.histogram_bin_edges(global_values, bins=bins)
    global_hist, _ = np.histogram(global_values, bins=bin_edges, density=True)

    sizes = eligible_clients.to_numpy(dtype=int)
    total_n = int(sizes.sum())

    rng = np.random.default_rng(seed)
    weighted_js_values: list[float] = []
    max_js_values: list[float] = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(global_values)
        start = 0
        js_vals: list[float] = []

        for client_size in sizes:
            stop = start + client_size
            client_values = shuffled[start:stop]
            start = stop

            js = _compute_histogram_js(client_values, bin_edges, global_hist)
            js_vals.append(js)

        js_arr = np.array(js_vals, dtype=float)
        weighted_js = float(np.average(js_arr, weights=sizes))
        max_js = float(np.max(js_arr))

        weighted_js_values.append(weighted_js)
        max_js_values.append(max_js)

    weighted_arr = np.array(weighted_js_values, dtype=float)
    max_arr = np.array(max_js_values, dtype=float)

    threshold = float(weighted_arr.mean() + 2.0 * weighted_arr.std(ddof=0))

    return {
        "num_clients_considered": int(len(sizes)),
        "num_samples_considered": total_n,
        "n_simulations": int(n_simulations),
        "weighted_js_null_mean": float(weighted_arr.mean()),
        "weighted_js_null_std": float(weighted_arr.std(ddof=0)),
        "weighted_js_null_p95": float(np.percentile(weighted_arr, 95)),
        "weighted_js_null_p99": float(np.percentile(weighted_arr, 99)),
        "max_js_null_mean": float(max_arr.mean()),
        "max_js_null_std": float(max_arr.std(ddof=0)),
        "js_threshold": threshold,
    }


def compute_non_iid_report(
    df: pd.DataFrame,
    client_col: str,
    target_col: str,
    bins: int,
    n_simulations: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    valid_df = df[[client_col, target_col]].dropna().copy()

    client_sizes = valid_df.groupby(client_col).size()
    eligible_clients = client_sizes[client_sizes >= 2].index
    valid_df = valid_df[valid_df[client_col].isin(eligible_clients)].copy()

    if valid_df.empty:
        raise ValueError("No client has at least 2 samples after filtering missing values.")

    global_values = valid_df[target_col].to_numpy(dtype=float)
    bin_edges = np.histogram_bin_edges(global_values, bins=bins)
    global_hist, _ = np.histogram(global_values, bins=bin_edges, density=True)

    client_rows: list[dict[str, float | str | int]] = []
    for client_id, cdf in valid_df.groupby(client_col):
        client_values = cdf[target_col].to_numpy(dtype=float)
        js = _compute_histogram_js(client_values, bin_edges, global_hist)
        client_rows.append(
            {
                "client_id": str(client_id),
                "num_samples": int(len(client_values)),
                "js_divergence_vs_global": js,
            }
        )

    js_df = (
        pd.DataFrame(client_rows)
        .sort_values("js_divergence_vs_global", ascending=False)
        .reset_index(drop=True)
    )

    weighted_js = float(
        np.average(
            js_df["js_divergence_vs_global"].to_numpy(dtype=float),
            weights=js_df["num_samples"].to_numpy(dtype=float),
        )
    )
    max_js = float(js_df["js_divergence_vs_global"].max())
    num_clients_above_threshold = 0

    iid_baseline = estimate_iid_js_baseline(
        valid_df,
        client_col=client_col,
        target_col=target_col,
        bins=bins,
        n_simulations=n_simulations,
        seed=random_seed,
    )
    js_threshold = float(iid_baseline["js_threshold"])
    num_clients_above_threshold = int((js_df["js_divergence_vs_global"] > js_threshold).sum())

    decision = "non-iid" if weighted_js > js_threshold else "approximately-iid"

    report = {
        "num_clients_considered": int(len(js_df)),
        "n_simulations": int(n_simulations),
        "weighted_js_divergence": weighted_js,
        "max_js_divergence": max_js,
        "js_threshold": js_threshold,
        "num_clients_above_threshold": num_clients_above_threshold,
        "decision": decision,
        **iid_baseline,
    }
    return js_df, report


def run_partition_eda(
    df: pd.DataFrame,
    partition_by: str,
    target_col: str,
    outdir: Path,
    bins: int,
    n_simulations: int,
    random_seed: int,
) -> None:
    if partition_by != "station":
        raise ValueError(f"Unsupported partition_by={partition_by}; only 'station' is supported.")

    client_col = "client_station"
    partition_dir = outdir / partition_by
    partition_dir.mkdir(parents=True, exist_ok=True)

    summary_df = compute_client_summary(df, client_col=client_col, target_col=target_col)
    summary_path = partition_dir / "client_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    js_df, non_iid_report = compute_non_iid_report(
        df=df,
        client_col=client_col,
        target_col=target_col,
        bins=bins,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )
    js_path = partition_dir / "non_iid_client_js.csv"
    js_df.to_csv(js_path, index=False)

    with (partition_dir / "non_iid_report.json").open("w", encoding="utf-8") as f:
        json.dump(non_iid_report, f, indent=2)

    js_threshold = float(non_iid_report["js_threshold"])

    plot_summary_distributions(
        summary_df,
        outpath=partition_dir / "summary_distributions.png",
        title=f"Federated EDA summary by {partition_by}",
    )
    plot_top_clients_boxplot(
        df,
        client_col=client_col,
        target_col=target_col,
        outpath=partition_dir / "top_clients_target_boxplot.png",
    )
    plot_js_distribution(
        js_df,
        outpath=partition_dir / "js_divergence_distribution.png",
        title=f"JS divergence distribution by {partition_by}",
        js_threshold=js_threshold,
    )
    plot_js_ranked_clients(
        js_df,
        outpath=partition_dir / "js_divergence_ranked_clients.png",
        title="",#f"Ranked client heterogeneity by {partition_by}",
        js_threshold=js_threshold,
    )
    plot_js_vs_samples(
        js_df,
        outpath=partition_dir / "js_divergence_vs_num_samples.png",
        title=f"Client heterogeneity vs sample count by {partition_by}",
        js_threshold=js_threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run federated EDA for station clients, including non-IID diagnostics."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to dataset_early_session.parquet",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outputs/federated_eda",
        help="Output directory for EDA artifacts",
    )
    parser.add_argument(
        "--site-name",
        type=str,
        default="caltech",
        help="Site name to filter before running EDA",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="kwh_delivered",
        help="Target column used for client statistics and non-IID analysis",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins for JS divergence computation",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=200,
        help="Number of IID null simulations used to estimate the dataset-specific JS threshold",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for IID baseline simulations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(data_path=args.data_path, site_name=args.site_name, cyclical=True)
    df = build_client_id_columns(df)

    if args.target_col not in df.columns:
        raise ValueError(
            f"target column '{args.target_col}' not found. Available columns: {list(df.columns)}"
        )

    run_partition_eda(
        df=df,
        partition_by="station",
        target_col=args.target_col,
        outdir=outdir,
        bins=args.bins,
        n_simulations=args.n_simulations,
        random_seed=args.random_seed,
    )

    print(f"Federated EDA complete. Artifacts saved under: {outdir.resolve()}")


if __name__ == "__main__":
    main()
