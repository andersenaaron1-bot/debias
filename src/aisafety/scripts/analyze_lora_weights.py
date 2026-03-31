"""Analyze LoRA adapters in weight space (no base model required).

This script summarizes *where* each LoRA adapter changes the base model by:
- computing Frobenius norms of the effective weight deltas (ΔW = s * B @ A)
  per adapted module (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj, etc.)
- optionally computing pairwise cosine similarity between adapters in ΔW-space.

It is useful as a lightweight "fingerprint" before doing activation-level analysis.

Example:
  python -m aisafety.scripts.analyze_lora_weights ^
    --adapter ai_tone=artifacts/lora/ai_tone ^
    --adapter academic_formality=artifacts/lora/academic_formality ^
    --adapter corporate_safety=artifacts/lora/corporate_safety ^
    --out-dir artifacts/lora_weight_analysis
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors


_LAYER_RE = re.compile(r"(?:^|\\.)layers\\.(\\d+)\\.")


def _parse_named_path(spec: str) -> tuple[str, Path]:
    spec = str(spec or "").strip()
    if not spec:
        raise ValueError("Empty adapter spec.")
    if "=" in spec:
        name, raw = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing name in adapter spec: {spec!r}")
        return name, Path(raw)
    path = Path(spec)
    return path.stem, path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_adapter_config(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")
    return _load_json(cfg_path)


def _load_adapter_state(adapter_dir: Path) -> dict[str, torch.Tensor]:
    st_path = adapter_dir / "adapter_model.safetensors"
    if st_path.exists():
        return load_safetensors(str(st_path), device="cpu")

    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.exists():
        obj = torch.load(bin_path, map_location="cpu")
        if not isinstance(obj, dict):
            raise TypeError(f"Unexpected object in {bin_path}: {type(obj)}")
        return obj

    raise FileNotFoundError(
        f"Missing adapter weights in {adapter_dir} (expected adapter_model.safetensors or adapter_model.bin)"
    )


def _split_lora_key(key: str) -> tuple[str, str] | None:
    """Return (prefix, kind) where kind is 'A' or 'B'."""
    if key.endswith(".lora_A.weight"):
        return key[: -len(".lora_A.weight")], "A"
    if key.endswith(".lora_B.weight"):
        return key[: -len(".lora_B.weight")], "B"
    if ".lora_A." in key and key.endswith(".weight"):
        prefix, _rest = key.split(".lora_A.", 1)
        return prefix, "A"
    if ".lora_B." in key and key.endswith(".weight"):
        prefix, _rest = key.split(".lora_B.", 1)
        return prefix, "B"
    return None


def _layer_idx_from_prefix(prefix: str) -> int | None:
    m = _LAYER_RE.search(prefix)
    if not m:
        return None
    return int(m.group(1))


def _module_name_from_prefix(prefix: str) -> str:
    return prefix.split(".")[-1]


@dataclass(frozen=True)
class LoraDelta:
    adapter: str
    prefix: str
    layer: int | None
    module: str
    scale: float
    A: torch.Tensor  # (r, in)
    B: torch.Tensor  # (out, r)

    @property
    def r(self) -> int:
        return int(self.A.shape[0])


def _iter_lora_deltas(adapter_name: str, cfg: dict, state: dict[str, torch.Tensor]) -> Iterable[LoraDelta]:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in state.items():
        split = _split_lora_key(str(k))
        if split is None:
            continue
        prefix, kind = split
        d = pairs.setdefault(prefix, {})
        d[kind] = v

    cfg_r = cfg.get("r")
    cfg_alpha = cfg.get("lora_alpha")

    for prefix, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        A = ab["A"]
        B = ab["B"]
        r = int(cfg_r) if cfg_r is not None else int(A.shape[0])
        alpha = float(cfg_alpha) if cfg_alpha is not None else float(r)
        scale = alpha / max(r, 1)
        yield LoraDelta(
            adapter=adapter_name,
            prefix=prefix,
            layer=_layer_idx_from_prefix(prefix),
            module=_module_name_from_prefix(prefix),
            scale=scale,
            A=A,
            B=B,
        )


def lora_fro_norm(delta: LoraDelta) -> float:
    """Compute ||ΔW||_F where ΔW = scale * (B @ A), without materializing ΔW."""
    A = delta.A.float()
    B = delta.B.float()
    # ||B A||_F^2 = trace(A^T B^T B A) = trace((B^T B) (A A^T))
    BtB = B.T @ B  # (r, r)
    AAt = A @ A.T  # (r, r)
    norm_sq = torch.trace(BtB @ AAt) * float(delta.scale**2)
    norm_sq = torch.clamp(norm_sq, min=0.0)
    return float(torch.sqrt(norm_sq).item())


def lora_inner_product(a: LoraDelta, b: LoraDelta) -> float:
    """Compute <ΔW_a, ΔW_b>_F without materializing either matrix."""
    if a.A.shape != b.A.shape or a.B.shape != b.B.shape:
        raise ValueError("ΔW inner product requires matching shapes (same module geometry).")
    A1 = a.A.float()
    B1 = a.B.float()
    A2 = b.A.float()
    B2 = b.B.float()
    K = B1.T @ B2  # (r, r)
    KA2 = K @ A2  # (r, in)
    inner = (A1 * KA2).sum() * float(a.scale * b.scale)
    return float(inner.item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="Adapter spec as name=PATH (repeatable). If name omitted, uses PATH stem.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/lora_weight_analysis"))
    p.add_argument(
        "--pairwise-similarity",
        action="store_true",
        help="Compute cosine similarity between adapters in ΔW-space.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.adapter:
        raise SystemExit("Provide at least one --adapter name=PATH.")

    adapters: list[tuple[str, Path]] = []
    for spec in args.adapter:
        name, path = _parse_named_path(spec)
        if not path.exists():
            raise FileNotFoundError(f"Adapter path not found: {path}")
        adapters.append((name, path))

    all_deltas: dict[str, dict[str, LoraDelta]] = {}
    rows = []
    for name, path in adapters:
        cfg = _load_adapter_config(path)
        state = _load_adapter_state(path)
        deltas = {d.prefix: d for d in _iter_lora_deltas(name, cfg, state)}
        all_deltas[name] = deltas

        for prefix, d in sorted(deltas.items(), key=lambda kv: (kv[1].layer is None, kv[1].layer, kv[1].module, kv[0])):
            n = lora_fro_norm(d)
            rows.append(
                {
                    "adapter": name,
                    "prefix": prefix,
                    "layer": d.layer,
                    "module": d.module,
                    "r": d.r,
                    "scale": d.scale,
                    "delta_fro_norm": n,
                }
            )

    df = pd.DataFrame(rows)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "module_norms.csv", index=False)

    # Convenience aggregates for plotting.
    if not df.empty:
        df.groupby(["adapter", "layer"], dropna=False)["delta_fro_norm"].sum().reset_index().to_csv(
            out_dir / "layer_norms.csv", index=False
        )
        df.groupby(["adapter", "layer", "module"], dropna=False)["delta_fro_norm"].sum().reset_index().to_csv(
            out_dir / "layer_module_norms.csv", index=False
        )

    if args.pairwise_similarity:
        sim_rows = []
        names = [n for n, _ in adapters]
        for i, a_name in enumerate(names):
            for b_name in names[i + 1 :]:
                a_mods = all_deltas[a_name]
                b_mods = all_deltas[b_name]
                shared = sorted(set(a_mods) & set(b_mods))
                if not shared:
                    sim_rows.append(
                        {"adapter_a": a_name, "adapter_b": b_name, "shared_modules": 0, "cosine": None}
                    )
                    continue
                dot = 0.0
                a_norm_sq = 0.0
                b_norm_sq = 0.0
                for prefix in shared:
                    da = a_mods[prefix]
                    db = b_mods[prefix]
                    na = lora_fro_norm(da)
                    nb = lora_fro_norm(db)
                    a_norm_sq += na * na
                    b_norm_sq += nb * nb
                    dot += lora_inner_product(da, db)
                denom = (a_norm_sq * b_norm_sq) ** 0.5
                cosine = dot / denom if denom else None
                sim_rows.append(
                    {
                        "adapter_a": a_name,
                        "adapter_b": b_name,
                        "shared_modules": len(shared),
                        "cosine": cosine,
                    }
                )

        pd.DataFrame(sim_rows).to_csv(out_dir / "adapter_cosine_similarity.csv", index=False)

    print(f"Wrote LoRA weight summaries to {out_dir}")


if __name__ == "__main__":
    main()

