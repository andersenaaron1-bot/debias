"""Print the claim-critical SummEval criterion validation readout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path


PRIMARY_METRICS = {
    "forced_target_adoption",
    "order_consistent_target_adoption",
    "order_consistent_rate",
    "natural_valid_rate",
    "phase2_budget_saturation_rate",
}
PRIMARY_CONTRASTS = {
    "criterion_scaffold_rescue",
    "criterion_scaffold_specificity",
    "generic_structure_effect",
    "score_evidence_rescue",
    "scaffold_gap_to_score_evidence",
    "explicit_target_vs_scaffold",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--activation-analysis-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--digits", type=int, default=3)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _print(title: str, frame: pd.DataFrame, *, digits: int) -> None:
    print(f"\n=== {title} ===")
    if frame.empty:
        print("NO ROWS")
    else:
        print(frame.round(int(digits)).to_string(index=False))


def read_results(
    root: Path,
    *,
    activation_root: Path | None,
    digits: int,
) -> None:
    endpoint = _read(root, "endpoint_summary.csv")
    if not endpoint.empty:
        endpoint = endpoint[
            endpoint["transition_type"].eq("all")
            & endpoint["metric"].isin(PRIMARY_METRICS)
        ]
    _print("SUMMEVAL 384-TOKEN ENDPOINTS", endpoint, digits=digits)

    effects = _read(root, "endpoint_effects.csv")
    if not effects.empty:
        effects = effects[
            effects["transition_type"].eq("all")
            & effects["metric"].isin(PRIMARY_METRICS)
            & effects["contrast"].isin(PRIMARY_CONTRASTS)
        ]
    _print("PAIR-BOOTSTRAP ENDPOINT EFFECTS", effects, digits=digits)

    by_transition = _read(root, "endpoint_summary.csv")
    if not by_transition.empty:
        by_transition = by_transition[
            by_transition["transition_type"].ne("all")
            & by_transition["metric"].isin(
                {
                    "forced_target_adoption",
                    "order_consistent_target_adoption",
                }
            )
        ]
    _print("ENDPOINTS BY TRANSITION STRATUM", by_transition, digits=digits)

    criterion_effects = _read(root, "criterion_endpoint_effects.csv")
    if not criterion_effects.empty:
        criterion_effects = criterion_effects[
            criterion_effects["transition_type"].eq("all")
            & criterion_effects["metric"].isin(
                {
                    "forced_target_adoption",
                    "order_consistent_target_adoption",
                    "order_consistent_rate",
                }
            )
            & criterion_effects["contrast"].isin(
                {
                    "criterion_scaffold_rescue",
                    "score_evidence_rescue",
                    "scaffold_gap_to_score_evidence",
                }
            )
        ][
            [
                "updated_criterion_id",
                "contrast",
                "left_condition",
                "right_condition",
                "metric",
                "n_pairs",
                "mean",
                "ci95_low",
                "ci95_high",
            ]
        ]
    _print("CRITERION-STRATIFIED EFFECTS", criterion_effects, digits=digits)

    checkpoints = _read(root, "checkpoint_summary.csv")
    if not checkpoints.empty:
        checkpoints = checkpoints[
            checkpoints["transition_type"].eq("all")
            & checkpoints["metric"].isin(
                {
                    "forced_target_adoption",
                    "order_consistent_target_adoption",
                }
            )
        ][
            [
                "condition_id",
                "stage",
                "budget_tokens",
                "metric",
                "n_pairs",
                "mean",
                "ci95_low",
                "ci95_high",
            ]
        ]
    _print("TARGET-ADOPTION CURVES", checkpoints, digits=digits)

    checkpoint_effects = _read(root, "checkpoint_effects.csv")
    if not checkpoint_effects.empty:
        checkpoint_effects = checkpoint_effects[
            checkpoint_effects["transition_type"].eq("all")
            & checkpoint_effects["contrast"].isin(
                {
                    "criterion_scaffold_rescue",
                    "criterion_scaffold_specificity",
                    "score_evidence_rescue",
                }
            )
            & checkpoint_effects["metric"].isin(
                {
                    "forced_target_adoption",
                    "order_consistent_target_adoption",
                }
            )
        ]
    _print("EFFECTS OVER TIME", checkpoint_effects, digits=digits)

    direct = _read(root, "direct_summary.csv")
    if not direct.empty:
        direct = direct[
            direct["transition_type"].eq("all")
            & direct["metric"].isin(PRIMARY_METRICS)
        ]
    _print("DIRECT-ANSWER BASELINE", direct, digits=digits)

    cot_direct = _read(root, "cot_vs_direct_effects.csv")
    if not cot_direct.empty:
        cot_direct = cot_direct[
            cot_direct["transition_type"].eq("all")
            & cot_direct["metric"].isin(
                {
                    "forced_target_adoption",
                    "order_consistent_target_adoption",
                    "order_consistent_rate",
                    "natural_valid_rate",
                }
            )
        ]
    _print("COT MINUS DIRECT", cot_direct, digits=digits)

    if activation_root is None:
        return
    activation = _read(
        activation_root,
        "point_metrics_by_condition.csv",
    )
    if not activation.empty:
        activation = activation[
            activation["point_name"].isin(
                {
                    "phase1_readout_128",
                    "phase2_readout_0",
                    "phase2_readout_384",
                }
            )
            & activation["probe_target"].isin(
                {
                    "active_criterion",
                    "criterion_target",
                    "current_choice",
                    "final_choice",
                }
            )
        ][
            [
                "condition_id",
                "probe_target",
                "hidden_layer",
                "point_name",
                "n_pairs",
                "balanced_accuracy",
                "macro_roc_auc",
            ]
        ]
    _print(
        "FIXED-LAYER ACTIVATION READOUTS BY CONDITION",
        activation,
        digits=digits,
    )


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    read_results(
        _resolve(workspace_root, args.analysis_dir),
        activation_root=(
            _resolve(workspace_root, args.activation_analysis_dir)
            if args.activation_analysis_dir is not None
            else None
        ),
        digits=int(args.digits),
    )


if __name__ == "__main__":
    main()
