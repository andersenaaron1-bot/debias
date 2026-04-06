"""Generate canonical J0/Jrepair experiment configs from a matrix spec."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--spec-json",
        type=Path,
        default=Path("configs/experiments/judge_repair_matrix_v1.json"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("configs/experiments"),
    )
    return p.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _slug(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _base_config(spec: dict) -> dict:
    return copy.deepcopy(spec["base_template"])


def _make_anchor_config(spec: dict) -> dict:
    cfg = _base_config(spec)
    anchor = spec["anchor_judge"]
    cfg["name"] = str(anchor["name"])
    cfg["description"] = str(anchor["description"])
    cfg["output_dir"] = str(anchor["output_dir"])
    for key, value in (anchor.get("top_level_overrides") or {}).items():
        cfg[key] = value
    cfg.setdefault("train_args", {})
    for key, value in (anchor.get("train_arg_overrides") or {}).items():
        cfg["train_args"][key] = value
    return cfg


def _make_full_repair_config(spec: dict) -> dict:
    cfg = _base_config(spec)
    full = spec["full_repair"]
    cfg["name"] = str(full["name"])
    cfg["description"] = str(full["description"])
    cfg["output_dir"] = str(full["output_dir"])
    return cfg


def _remove_from_list(values: list[str], target: str) -> list[str]:
    return [v for v in values if str(v) != str(target)]


def _cue_family_ablation_config(spec: dict, family: str) -> dict:
    cfg = _make_full_repair_config(spec)
    family_slug = _slug(family)
    cfg["name"] = f"jrepair_loo_cue_{family_slug}_v1"
    cfg["description"] = (
        "Full repair judge with cue-adversarial removal active except for "
        f"{family}."
    )
    cfg["output_dir"] = f"artifacts/reward/{cfg['name']}"
    current = list(cfg["train_args"]["cue_families"])
    cfg["train_args"]["cue_families"] = _remove_from_list(current, family)
    return cfg


def _axis_ablation_config(spec: dict, axis: str) -> dict:
    cfg = _make_full_repair_config(spec)
    axis_slug = _slug(axis)
    cfg["name"] = f"jrepair_loo_axis_{axis_slug}_v1"
    cfg["description"] = (
        "Full repair judge with paired invariance active except for "
        f"the {axis} style axis."
    )
    cfg["output_dir"] = f"artifacts/reward/{cfg['name']}"
    cfg["train_args"]["exclude_axes"] = [axis]
    return cfg


def _joint_ablation_config(spec: dict, payload: dict) -> dict:
    cfg = _make_full_repair_config(spec)
    joint_name = str(payload["joint_name"])
    cfg["name"] = f"jrepair_loo_joint_{_slug(joint_name)}_v1"
    cfg["description"] = (
        "Full repair judge with both paired invariance and cue-adversarial "
        f"removal excluding {joint_name}."
    )
    cfg["output_dir"] = f"artifacts/reward/{cfg['name']}"
    cfg["style_train_jsonl"] = str(payload["style_train_jsonl"])
    cfg["style_val_jsonl"] = str(payload["style_val_jsonl"])
    current = list(cfg["train_args"]["cue_families"])
    cfg["train_args"]["cue_families"] = _remove_from_list(current, str(payload["cue_family"]))
    return cfg


def main() -> None:
    args = parse_args()
    spec = _load_json(Path(args.spec_json))
    out_dir = Path(args.out_dir)

    configs: list[dict] = []
    configs.append(_make_anchor_config(spec))
    configs.append(_make_full_repair_config(spec))
    for family in spec.get("cue_family_leave_one_out") or []:
        configs.append(_cue_family_ablation_config(spec, str(family)))
    for axis in spec.get("invariance_axis_leave_one_out") or []:
        configs.append(_axis_ablation_config(spec, str(axis)))
    for payload in spec.get("joint_ablation_map") or []:
        configs.append(_joint_ablation_config(spec, dict(payload)))

    for cfg in configs:
        _write_json(out_dir / f"{cfg['name']}.json", cfg)

    summary = {
        "spec_json": str(Path(args.spec_json)),
        "out_dir": str(out_dir),
        "generated": [f"{cfg['name']}.json" for cfg in configs],
    }
    _write_json(out_dir / "judge_repair_matrix_v1.generated.json", summary)
    print(f"Generated {len(configs)} experiment configs in {out_dir}")


if __name__ == "__main__":
    main()
