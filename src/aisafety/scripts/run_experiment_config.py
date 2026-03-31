"""Launch a training run from a JSON experiment config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


BOOL_FLAGS = {
    "use_4bit": "--use-4bit",
    "bf16": "--bf16",
    "fp16": "--fp16",
    "gradient_checkpointing": "--gradient-checkpointing",
}

LIST_AS_CSV = {"anchor_attribute_names", "cue_families"}

PATH_KEYS = {
    "pref_train_jsonl",
    "pref_val_jsonl",
    "anchor_train_jsonl",
    "anchor_val_jsonl",
    "style_train_jsonl",
    "style_val_jsonl",
    "cue_train_jsonl",
    "cue_val_jsonl",
    "output_dir",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workspace-root", type=Path, default=Path.cwd())
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--print-only", action="store_true")
    return p.parse_args()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Experiment config at {path} must be a JSON object.")
    return payload


def _resolve_path(value: str, *, workspace_root: Path) -> str:
    p = Path(str(value))
    if p.is_absolute():
        return str(p)
    return str((workspace_root / p).resolve())


def _append_arg(cmd: list[str], key: str, value, *, workspace_root: Path) -> None:
    cli_key = f"--{key.replace('_', '-')}"
    if key in BOOL_FLAGS:
        if bool(value):
            cmd.append(BOOL_FLAGS[key])
        return
    if value is None:
        return
    if key in LIST_AS_CSV:
        if not value:
            return
        if not isinstance(value, list):
            raise TypeError(f"{key} must be a list in experiment config.")
        cmd.extend([cli_key, ",".join(str(v) for v in value)])
        return
    if key in PATH_KEYS:
        cmd.extend([cli_key, _resolve_path(str(value), workspace_root=workspace_root)])
        return
    cmd.extend([cli_key, str(value)])


def build_train_command(config: dict, *, workspace_root: Path, output_dir_override: Path | None) -> list[str]:
    train_args = dict(config.get("train_args") or {})
    cmd = [sys.executable, "-m", "aisafety.scripts.train_reward_lora"]

    top_level_order = [
        "model_id",
        "pref_train_jsonl",
        "pref_val_jsonl",
        "anchor_train_jsonl",
        "anchor_val_jsonl",
        "style_train_jsonl",
        "style_val_jsonl",
        "cue_train_jsonl",
        "cue_val_jsonl",
        "output_dir",
    ]
    for key in top_level_order:
        value = config.get(key)
        if key == "output_dir" and output_dir_override is not None:
            value = str(output_dir_override)
        _append_arg(cmd, key, value, workspace_root=workspace_root)

    for key, value in train_args.items():
        _append_arg(cmd, key, value, workspace_root=workspace_root)

    return cmd


def main() -> None:
    args = parse_args()
    config = _load_config(Path(args.config))
    cmd = build_train_command(
        config,
        workspace_root=Path(args.workspace_root),
        output_dir_override=None if args.output_dir is None else Path(args.output_dir),
    )
    print(" ".join(cmd))
    if not bool(args.print_only):
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
