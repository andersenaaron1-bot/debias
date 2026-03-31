"""Pivot reward-adapter fingerprint CSVs into plot-friendly matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _write_matrix(df: pd.DataFrame, *, index: str, columns: str, values: str, out_path: Path) -> None:
    mat = df.pivot(index=index, columns=columns, values=values)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat.to_csv(out_path)


def _pairwise_mean_matrix(df: pd.DataFrame, *, a: str, b: str, value: str) -> pd.DataFrame:
    mean = df.groupby([a, b], dropna=False)[value].mean().reset_index()
    names = sorted(set(mean[a].astype(str)) | set(mean[b].astype(str)))
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for n in names:
        mat.loc[n, n] = 1.0
    for r in mean.itertuples(index=False):
        ra = str(getattr(r, a))
        rb = str(getattr(r, b))
        v = float(getattr(r, value))
        mat.loc[ra, rb] = v
        mat.loc[rb, ra] = v
    return mat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Directory produced by a reward-adapter fingerprint scan.",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Output dir (defaults to --in-dir).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir or in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lens_path = in_dir / "logit_lens_by_layer.csv"
    if lens_path.exists():
        df = pd.read_csv(lens_path)
        _write_matrix(
            df,
            index="layer",
            columns="run",
            values="logit_lens_mean",
            out_path=out_dir / "logit_lens_mean_matrix.csv",
        )
        _write_matrix(
            df,
            index="layer",
            columns="run",
            values="logit_lens_std",
            out_path=out_dir / "logit_lens_std_matrix.csv",
        )

        mat = df.pivot(index="layer", columns="run", values="logit_lens_mean")
        if "base" in mat.columns and len(mat.columns) > 1:
            delta = mat.subtract(mat["base"], axis=0).drop(columns=["base"])
            delta.to_csv(out_dir / "logit_lens_mean_delta_vs_base_matrix.csv")

    delta_path = in_dir / "delta_hidden_mean_norm_by_layer.csv"
    if delta_path.exists():
        df = pd.read_csv(delta_path)
        _write_matrix(df, index="layer", columns="run", values="delta_hidden_mean_norm", out_path=out_dir / "delta_hidden_mean_norm_matrix.csv")

    sim_hidden = in_dir / "adapter_similarity_hidden.csv"
    if sim_hidden.exists():
        df = pd.read_csv(sim_hidden)
        mat = _pairwise_mean_matrix(df, a="run_a", b="run_b", value="cosine")
        mat.to_csv(out_dir / "adapter_similarity_hidden_mean_over_layers.csv")

    sim_sae = in_dir / "adapter_similarity_sae.csv"
    if sim_sae.exists():
        df = pd.read_csv(sim_sae)
        mat = _pairwise_mean_matrix(df, a="run_a", b="run_b", value="cosine")
        mat.to_csv(out_dir / "adapter_similarity_sae_mean_over_layers.csv")

    print(f"Wrote pivoted matrices to {out_dir}")


if __name__ == "__main__":
    main()
