"""Build unified style-variant meaning groups for invariance training.

This script standardizes multiple parallel rewrite corpora into a single schema:
  group_id, style_axis, domain, variants[], source_dataset, meta

Filtering (default):
  - numbers consistency vs anchor variant
  - embedding cosine similarity vs anchor (MiniLM) with threshold

Example:
  python -m aisafety.scripts.build_style_groups_hf ^
    --config data\\derived\\style_groups\\style_axes_config.json ^
    --out-dir data\\derived\\style_groups ^
    --allow-trust-remote-code
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

import torch
import torch.nn.functional as F
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from aisafety.config import DATA_DIR, DEFAULT_CACHE_DIR, DEFAULT_SEED


NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")

# Minimal explicit-threat heuristics for ParaDetox to reduce "meaning change" confounds.
# This avoids maintaining a raw slur list; identity-hate/slur filtering is handled
# via an optional lightweight toxicity classifier (see --paradetox-risk-filter).
_PARADETOX_THREAT_PATTERNS = (
    r"\b(?:i\s*(?:will|'ll|am\s+going\s+to|gonna))\s+(?:fucking\s+)?(?:kill|murder|shoot|stab|hang|strangle)\b",
    r"\b(?:kill|murder|shoot|stab)\s+(?:you|u|him|her|them|ya)\b",
    r"\b(?:kill|shoot)\s+yourself\b",
    r"\b(?:rape|raping)\s+(?:you|u|him|her|them|ya)\b",
    r"\b(?:i\s*(?:will|'ll|am\s+going\s+to|gonna))\s+(?:hurt|beat)\s+(?:you|u)\b",
)
_PARADETOX_THREAT_RE = re.compile("|".join(_PARADETOX_THREAT_PATTERNS), flags=re.IGNORECASE)


def _paradetox_has_explicit_threat(text: str) -> bool:
    return bool(_PARADETOX_THREAT_RE.search(str(text or "")))


class ParaDetoxRiskModel:
    """Multi-label toxicity model used to drop threat/identity-hate cases from ParaDetox."""

    def __init__(self, *, model_id: str, cache_dir: Path, device: str):
        self.model_id = str(model_id)
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(self.model_id, cache_dir=str(cache_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, cache_dir=str(cache_dir))
        self.model.to(self.device)
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", None)
        if not isinstance(id2label, dict) or not id2label:
            raise ValueError(f"Risk model {self.model_id!r} is missing id2label; cannot map threat/identity.")
        labels = {int(i): str(l).lower() for i, l in id2label.items()}
        inv = {l: i for i, l in labels.items()}

        def find_idx(substr: str) -> int | None:
            for lab, idx in inv.items():
                if substr in lab:
                    return int(idx)
            return None

        self.threat_idx = find_idx("threat")
        self.identity_idx = find_idx("identity") or find_idx("identity_hate")
        if self.threat_idx is None or self.identity_idx is None:
            raise ValueError(
                f"Risk model {self.model_id!r} labels do not include 'threat' and 'identity*'. "
                f"Found: {sorted(inv.keys())[:20]}..."
            )

    @torch.no_grad()
    def score(self, texts: list[str], *, batch_size: int = 64) -> tuple[list[float], list[float]]:
        threat: list[float] = []
        identity: list[float] = []
        for i in range(0, len(texts), int(batch_size)):
            batch = [str(t) for t in texts[i : i + int(batch_size)]]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc, return_dict=True)
            probs = torch.sigmoid(out.logits).detach().float().cpu()
            threat.extend(probs[:, int(self.threat_idx)].tolist())
            identity.extend(probs[:, int(self.identity_idx)].tolist())
        return threat, identity


@dataclass(frozen=True)
class StyleSourceSpec:
    style_axis: str
    source_dataset: str
    domain: str
    loader: str
    dataset_id: str | None = None
    config_name: str | None = None
    hf_revision: str | None = None
    splits: list[str] | None = None
    split_name_contains: list[str] | None = None
    streaming: bool = True
    shuffle: bool = True
    shuffle_buffer_size: int = 50_000
    max_groups: int | None = None
    # Optional override for the embedding similarity threshold. This is useful for
    # "distant style" axes (e.g., archaic register) where lexical divergence can
    # depress cosine similarity even when meaning is preserved.
    sim_threshold: float | None = None
    # GitHub archive loader options (used for datasets that are not on HF).
    github_repo: str | None = None  # e.g. "harsh19/Shakespearizing-Modern-English"
    github_revision: str | None = None  # e.g. "master" or a commit SHA
    github_subdir: str | None = None  # e.g. "data"
    github_src_token: str | None = None  # e.g. "modern"
    github_tgt_token: str | None = None  # e.g. "original"
    trust_remote_code: bool = False
    src_col: str | None = None
    tgt_col: str | None = None
    refs_col: str | None = None
    texts_col: str | None = None
    tgz_files: list[str] | None = None


@dataclass(frozen=True)
class GroupCandidate:
    style_axis: str
    source_dataset: str
    domain: str
    variants: list[str]
    meta: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=DATA_DIR / "derived" / "style_groups" / "style_axes_config.json",
        help="JSON config listing datasets and column mappings.",
    )
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / "derived" / "style_groups")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--apply-number-filter", action="store_true")
    p.add_argument("--no-number-filter", dest="apply_number_filter", action="store_false")
    p.set_defaults(apply_number_filter=True)
    p.add_argument("--apply-embed-filter", action="store_true")
    p.add_argument("--no-embed-filter", dest="apply_embed_filter", action="store_false")
    p.set_defaults(apply_embed_filter=True)
    p.add_argument("--embed-model-id", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embed-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sim-threshold", type=float, default=0.78)
    p.add_argument("--embed-batch-groups", type=int, default=64, help="Groups per embedding-filter batch.")
    p.add_argument("--embed-text-batch-size", type=int, default=256, help="Texts per embedder forward pass.")
    p.add_argument(
        "--allow-trust-remote-code",
        action="store_true",
        help="Allow datasets requiring trust_remote_code=True (e.g., GEM/BiSECT).",
    )
    p.add_argument("--max-groups-per-axis", type=int, default=None, help="Optional cap per style_axis.")
    p.add_argument("--paradetox-threat-regex-filter", action="store_true")
    p.add_argument("--no-paradetox-threat-regex-filter", dest="paradetox_threat_regex_filter", action="store_false")
    p.set_defaults(paradetox_threat_regex_filter=True)
    p.add_argument(
        "--paradetox-risk-filter",
        action="store_true",
        help="Drop ParaDetox pairs containing high-probability threat/identity-hate (uses a small classifier).",
    )
    p.add_argument("--no-paradetox-risk-filter", dest="paradetox_risk_filter", action="store_false")
    p.set_defaults(paradetox_risk_filter=True)
    p.add_argument("--paradetox-risk-model-id", type=str, default="unitary/unbiased-toxic-roberta")
    p.add_argument("--paradetox-risk-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--paradetox-risk-batch-size", type=int, default=64)
    p.add_argument("--paradetox-threat-threshold", type=float, default=0.35)
    p.add_argument("--paradetox-identity-threshold", type=float, default=0.35)
    return p.parse_args()


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\r\n", "\n").replace("\r", "\n")).strip()


def _dedup_variants(variants: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        v = _norm_text(v)
        if not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _numbers_multiset(text: str) -> Counter[str]:
    return Counter(NUM_RE.findall(text))


def _passes_number_filter(anchor: str, variants: list[str]) -> bool:
    a = _numbers_multiset(anchor)
    for v in variants:
        if _numbers_multiset(v) != a:
            return False
    return True


def _assign_split(group_id: str, *, seed: int, train_frac: float, val_frac: float) -> str:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1)")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0,1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")
    h = _sha1_hex(f"{seed}:{group_id}")
    r = int(h[:8], 16) / float(2**32)
    if r < train_frac:
        return "train"
    if r < train_frac + val_frac:
        return "val"
    return "test"


def _compute_group_id(style_axis: str, source_dataset: str, variants: list[str]) -> str:
    payload = {
        "style_axis": str(style_axis),
        "source_dataset": str(source_dataset),
        "variants": sorted(variants),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha1_hex(raw)


class TextEmbedder:
    def __init__(self, *, model_id: str, cache_dir: Path, device: str):
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model = AutoModel.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: list[str], *, batch_size: int) -> torch.Tensor:
        outs: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc, return_dict=True)
            last_hidden = out.last_hidden_state  # [B,T,H]
            mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = F.normalize(pooled, p=2, dim=1)
            outs.append(pooled.detach().cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty((0, 1))


def _read_config(path: Path) -> list[StyleSourceSpec]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    axes = raw.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("Config JSON must contain a non-empty 'axes' list.")
    out: list[StyleSourceSpec] = []
    for a in axes:
        if not isinstance(a, dict):
            raise TypeError("Each axes entry must be a dict.")
        out.append(StyleSourceSpec(**a))
    return out


def _load_hf_dataset(spec: StyleSourceSpec, *, split: str | None, allow_trust_remote_code: bool):
    if spec.trust_remote_code and not allow_trust_remote_code:
        raise RuntimeError(
            f"{spec.dataset_id} requires trust_remote_code=True; rerun with --allow-trust-remote-code"
        )
    return load_dataset(
        str(spec.dataset_id),
        spec.config_name,
        split=split,
        streaming=bool(spec.streaming),
        revision=None if spec.hf_revision is None else str(spec.hf_revision),
        trust_remote_code=bool(spec.trust_remote_code),
    )


def _iter_hf_rows(spec: StyleSourceSpec, *, seed: int, allow_trust_remote_code: bool) -> Iterator[tuple[str, dict]]:
    splits = spec.splits or []
    if not splits:
        ds_any = _load_hf_dataset(spec, split=None, allow_trust_remote_code=allow_trust_remote_code)
        if not isinstance(ds_any, (DatasetDict, IterableDatasetDict)):
            raise TypeError(f"Expected a dataset dict for split=None, got {type(ds_any).__name__}")
        for split_name, ds in ds_any.items():
            if spec.split_name_contains:
                sn = str(split_name).lower()
                if not any(s.lower() in sn for s in spec.split_name_contains):
                    continue
            yield from _iter_named_split_rows(ds, split_name, spec, seed=seed)
        return

    for split_name in splits:
        try:
            ds = _load_hf_dataset(spec, split=str(split_name), allow_trust_remote_code=allow_trust_remote_code)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] Skipping {spec.dataset_id} split={split_name!r}: {exc}")
            continue
        yield from _iter_named_split_rows(ds, str(split_name), spec, seed=seed)


def _iter_named_split_rows(ds, split_name: str, spec: StyleSourceSpec, *, seed: int) -> Iterator[tuple[str, dict]]:
    if bool(spec.shuffle):
        if getattr(ds, "_is_streaming", False):  # IterableDataset
            ds = ds.shuffle(seed=int(seed), buffer_size=int(spec.shuffle_buffer_size))
        else:
            ds = ds.shuffle(seed=int(seed))
    for row in ds:
        if not isinstance(row, dict):
            continue
        yield split_name, row


def _iter_groups_hf_pairs(spec: StyleSourceSpec, *, seed: int, allow_trust_remote_code: bool) -> Iterator[GroupCandidate]:
    if not spec.src_col or not spec.tgt_col:
        raise ValueError(f"{spec.source_dataset} hf_pairs requires src_col and tgt_col")
    for split_name, row in _iter_hf_rows(spec, seed=seed, allow_trust_remote_code=allow_trust_remote_code):
        src = row.get(spec.src_col)
        tgt = row.get(spec.tgt_col)
        if not isinstance(src, str):
            continue
        variants_raw: list[str] = [src]
        if isinstance(tgt, str):
            variants_raw.append(tgt)
        elif isinstance(tgt, list):
            variants_raw.extend([t for t in tgt if isinstance(t, str)])
        else:
            continue
        variants = _dedup_variants(variants_raw)
        if len(variants) < 2:
            continue
        meta = {"hf_split": split_name}
        yield GroupCandidate(
            style_axis=str(spec.style_axis),
            source_dataset=str(spec.source_dataset),
            domain=str(spec.domain),
            variants=variants,
            meta=meta,
        )


def _iter_groups_hf_multiref(
    spec: StyleSourceSpec, *, seed: int, allow_trust_remote_code: bool
) -> Iterator[GroupCandidate]:
    if not spec.src_col or not spec.refs_col:
        raise ValueError(f"{spec.source_dataset} hf_multiref requires src_col and refs_col")
    for split_name, row in _iter_hf_rows(spec, seed=seed, allow_trust_remote_code=allow_trust_remote_code):
        src = row.get(spec.src_col)
        refs = row.get(spec.refs_col)
        if not isinstance(src, str):
            continue
        ref_list: list[str] = []
        if isinstance(refs, list):
            ref_list = [r for r in refs if isinstance(r, str)]
        variants = _dedup_variants([src, *ref_list])
        if len(variants) < 2:
            continue
        meta = {"hf_split": split_name}
        yield GroupCandidate(
            style_axis=str(spec.style_axis),
            source_dataset=str(spec.source_dataset),
            domain=str(spec.domain),
            variants=variants,
            meta=meta,
        )


def _iter_groups_hf_texts_list(
    spec: StyleSourceSpec, *, seed: int, allow_trust_remote_code: bool
) -> Iterator[GroupCandidate]:
    if not spec.texts_col:
        raise ValueError(f"{spec.source_dataset} hf_texts_list requires texts_col")
    for split_name, row in _iter_hf_rows(spec, seed=seed, allow_trust_remote_code=allow_trust_remote_code):
        texts = row.get(spec.texts_col)
        if not isinstance(texts, list) or len(texts) < 2:
            continue
        a, b = texts[0], texts[1]
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        variants = _dedup_variants([a, b])
        if len(variants) < 2:
            continue
        meta = {"hf_split": split_name, "n_texts": len(texts)}
        yield GroupCandidate(
            style_axis=str(spec.style_axis),
            source_dataset=str(spec.source_dataset),
            domain=str(spec.domain),
            variants=variants,
            meta=meta,
        )


def _iter_groups_gyafc_style_transfer_tgz(spec: StyleSourceSpec, *, cache_dir: Path) -> Iterator[GroupCandidate]:
    tgz_files = spec.tgz_files or []
    if not tgz_files:
        raise ValueError("gyafc_style_transfer_tgz requires tgz_files")

    for tgz_name in tgz_files:
        tgz_path = Path(
            hf_hub_download(
                repo_id=str(spec.source_dataset),
                repo_type="dataset",
                filename=str(tgz_name),
                cache_dir=str(cache_dir),
            )
        )
        extract_root = cache_dir / "gyafc_style_transfer" / _sha1_hex(str(tgz_path))
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz_path, "r:gz") as tf:
                tf.extractall(path=extract_root)

        txt_files = sorted([p for p in extract_root.rglob("*") if p.is_file()])
        informal: dict[str, list[Path]] = defaultdict(list)
        formal: dict[str, list[Path]] = defaultdict(list)
        for p in txt_files:
            name = p.name.lower()
            split = None
            for s in ("train", "valid", "validation", "dev", "test", "tune"):
                if s in name or s in str(p.parent).lower():
                    split = "validation" if s in {"valid", "validation", "dev", "tune"} else s
                    break
            if split is None:
                continue
            # RUCAIBox/Style-Transfer uses both (informal/formal) and (.src/.tgt) conventions.
            # For GYAFC we treat src=informal and tgt=formal.
            if "informal" in name or name.endswith(".src"):
                informal[split].append(p)
            elif "formal" in name or name.endswith(".tgt"):
                formal[split].append(p)

        for split, inf_paths in informal.items():
            form_paths = formal.get(split) or []
            if not inf_paths or not form_paths:
                continue
            inf_path = sorted(inf_paths)[0]
            form_paths = sorted(form_paths)
            inf_lines = inf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            form_lines_by_file = [
                fp.read_text(encoding="utf-8", errors="ignore").splitlines() for fp in form_paths
            ]
            n = min([len(inf_lines), *[len(x) for x in form_lines_by_file]])
            for i in range(n):
                src = inf_lines[i].strip()
                refs = [x[i].strip() for x in form_lines_by_file]
                variants = _dedup_variants([src, *refs])
                if len(variants) < 2:
                    continue
                meta = {"source_file": str(tgz_name), "split": split, "row": i}
                yield GroupCandidate(
                    style_axis=str(spec.style_axis),
                    source_dataset=str(spec.source_dataset),
                    domain=str(spec.domain),
                    variants=variants,
                    meta=meta,
                )


def _github_archive_url(repo: str, revision: str) -> str:
    rev = str(revision).strip()
    repo = str(repo).strip().removesuffix("/")
    if repo.startswith("https://github.com/"):
        repo = repo.removeprefix("https://github.com/").strip("/")
    if rev.startswith("refs/"):
        return f"https://github.com/{repo}/archive/{rev}.zip"
    # Heuristic: if not a full ref, treat it as a branch name.
    return f"https://github.com/{repo}/archive/refs/heads/{rev}.zip"


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(dest)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _pair_tokenized_parallel_files(
    files: list[Path], *, src_token: str, tgt_tokens: list[str]
) -> list[tuple[str, Path, Path]]:
    def key_for(name: str, token: str) -> str | None:
        parts = name.split(".")
        if token not in parts:
            return None
        # Remove exactly one occurrence of token.
        removed = False
        out = []
        for p in parts:
            if not removed and p == token:
                removed = True
                continue
            out.append(p)
        return ".".join(out)

    src_by: dict[str, Path] = {}
    tgt_by: dict[str, Path] = {}
    for p in files:
        k = key_for(p.name, src_token)
        if k is not None:
            src_by[k] = p
            continue
        for tok in tgt_tokens:
            k2 = key_for(p.name, tok)
            if k2 is not None:
                tgt_by[k2] = p
                break

    pairs: list[tuple[str, Path, Path]] = []
    for k in sorted(set(src_by) & set(tgt_by)):
        pairs.append((k, src_by[k], tgt_by[k]))
    return pairs


def _iter_groups_github_parallel_archive(spec: StyleSourceSpec, *, cache_dir: Path) -> Iterator[GroupCandidate]:
    if not spec.github_repo or not spec.github_revision:
        raise ValueError("github_parallel_archive requires github_repo and github_revision")

    repo = str(spec.github_repo)
    revision = str(spec.github_revision)
    subdir = str(spec.github_subdir or "data").strip().strip("/")
    src_token = str(spec.github_src_token or "modern").strip()
    tgt_tokens = [str(spec.github_tgt_token or "original").strip(), "original", "shakespearean", "shakespeare"]
    # De-dupe while preserving order.
    seen_tok: set[str] = set()
    tgt_tokens = [t for t in tgt_tokens if t and not (t in seen_tok or seen_tok.add(t))]

    zip_url = _github_archive_url(repo, revision)
    zip_path = cache_dir / "github_archives" / repo.replace("/", "--") / f"{revision}.zip"
    _download_url(zip_url, zip_path)

    extract_root = cache_dir / "github_archives" / repo.replace("/", "--") / f"{revision}.unpacked"
    if not extract_root.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=extract_root)

    # The archive usually contains a single top-level folder.
    top_dirs = [p for p in extract_root.iterdir() if p.is_dir()]
    if not top_dirs:
        raise RuntimeError(f"No folders found after extracting {zip_path}")
    repo_root = top_dirs[0] if len(top_dirs) == 1 else extract_root

    data_dir = repo_root / subdir
    if not data_dir.exists():
        raise RuntimeError(f"Expected subdir {subdir!r} inside GitHub archive, not found at {data_dir}")

    files = [p for p in data_dir.rglob("*") if p.is_file()]
    pairs = _pair_tokenized_parallel_files(files, src_token=src_token, tgt_tokens=tgt_tokens)
    if not pairs:
        raise RuntimeError(
            f"Could not find any parallel file pairs under {data_dir} using src_token={src_token!r} "
            f"and tgt_tokens={tgt_tokens!r}"
        )

    for key, src_path, tgt_path in pairs:
        src_lines = src_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        tgt_lines = tgt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        n = min(len(src_lines), len(tgt_lines))
        for i in range(n):
            src = src_lines[i].strip()
            tgt = tgt_lines[i].strip()
            variants = _dedup_variants([src, tgt])
            if len(variants) < 2:
                continue
            meta = {
                "github_repo": repo,
                "github_revision": revision,
                "pair_key": key,
                "src_file": str(src_path.relative_to(repo_root)),
                "tgt_file": str(tgt_path.relative_to(repo_root)),
                "row": i,
            }
            yield GroupCandidate(
                style_axis=str(spec.style_axis),
                source_dataset=str(spec.source_dataset),
                domain=str(spec.domain),
                variants=variants,
                meta=meta,
            )


def _iter_groups_for_spec(
    spec: StyleSourceSpec, *, seed: int, cache_dir: Path, allow_trust_remote_code: bool
) -> Iterator[GroupCandidate]:
    if spec.loader == "hf_pairs":
        yield from _iter_groups_hf_pairs(spec, seed=seed, allow_trust_remote_code=allow_trust_remote_code)
        return
    if spec.loader == "hf_multiref":
        yield from _iter_groups_hf_multiref(spec, seed=seed, allow_trust_remote_code=allow_trust_remote_code)
        return
    if spec.loader == "hf_texts_list":
        yield from _iter_groups_hf_texts_list(spec, seed=seed, allow_trust_remote_code=allow_trust_remote_code)
        return
    if spec.loader == "gyafc_style_transfer_tgz":
        yield from _iter_groups_gyafc_style_transfer_tgz(spec, cache_dir=cache_dir)
        return
    if spec.loader == "github_parallel_archive":
        yield from _iter_groups_github_parallel_archive(spec, cache_dir=cache_dir)
        return
    raise ValueError(f"Unknown loader: {spec.loader!r}")


def _filter_and_write(
    groups: Iterable[GroupCandidate],
    *,
    embedder: TextEmbedder | None,
    paradetox_risk_model: ParaDetoxRiskModel | None,
    paradetox_risk_batch_size: int,
    paradetox_threat_threshold: float,
    paradetox_identity_threshold: float,
    apply_number_filter: bool,
    apply_embed_filter: bool,
    sim_threshold: float,
    embed_batch_groups: int,
    embed_text_batch_size: int,
    paradetox_threat_regex_filter: bool,
    seed: int,
    train_frac: float,
    val_frac: float,
    out_files: dict[str, Any],
    seen_ids: set[str],
    max_groups_per_axis: int | None,
    counts: dict[str, Counter[str]],
) -> None:
    buffer: list[GroupCandidate] = []

    def flush(buf: list[GroupCandidate]) -> None:
        if not buf:
            return
        # ParaDetox risk filter (drop threats + identity-hate) before embedding work.
        if paradetox_risk_model is not None:
            detox_idx = [i for i, g in enumerate(buf) if g.style_axis == "detox_tone"]
            if detox_idx:
                detox_texts = [buf[i].variants[0] for i in detox_idx]
                thr, ident = paradetox_risk_model.score(
                    detox_texts, batch_size=int(paradetox_risk_batch_size)
                )
                to_drop = set()
                for i, t, ih in zip(detox_idx, thr, ident, strict=True):
                    if float(t) >= float(paradetox_threat_threshold) or float(ih) >= float(
                        paradetox_identity_threshold
                    ):
                        to_drop.add(i)
                        counts["detox_tone"]["risk_model_dropped"] += 1
                if to_drop:
                    buf = [g for i, g in enumerate(buf) if i not in to_drop]
                    if not buf:
                        return

        if apply_embed_filter:
            if embedder is None:
                raise RuntimeError("Embedding filter enabled but embedder is None")
            anchors = [g.variants[0] for g in buf]
            cand_texts: list[str] = []
            slices: list[tuple[int, int]] = []
            for g in buf:
                start = len(cand_texts)
                cand_texts.extend(g.variants[1:])
                end = len(cand_texts)
                slices.append((start, end))
            emb_a = embedder.embed_texts(anchors, batch_size=embed_text_batch_size)
            emb_c = embedder.embed_texts(cand_texts, batch_size=embed_text_batch_size) if cand_texts else None
        else:
            emb_a = None
            emb_c = None
            slices = []

        for i, g in enumerate(buf):
            counts[g.style_axis]["seen"] += 1
            if max_groups_per_axis is not None and counts[g.style_axis]["kept"] >= max_groups_per_axis:
                counts[g.style_axis]["cap_skipped"] += 1
                continue

            variants = g.variants
            if apply_embed_filter:
                anchor = variants[0]
                start, end = slices[i]
                kept: list[str] = []
                if emb_c is not None and end > start:
                    a_vec = emb_a[i : i + 1]  # [1,H]
                    c_vec = emb_c[start:end]  # [K,H]
                    sims = (c_vec @ a_vec.T).squeeze(1)
                    for v, s in zip(variants[1:], sims.tolist(), strict=True):
                        if float(s) >= sim_threshold:
                            kept.append(v)
                variants = _dedup_variants([anchor, *kept])
                if len(variants) < 2:
                    counts[g.style_axis]["embed_dropped"] += 1
                    continue

            gid = _compute_group_id(g.style_axis, g.source_dataset, variants)
            if gid in seen_ids:
                counts[g.style_axis]["dedup_dropped"] += 1
                continue
            seen_ids.add(gid)

            split = _assign_split(gid, seed=seed, train_frac=train_frac, val_frac=val_frac)
            out = {
                "group_id": gid,
                "style_axis": g.style_axis,
                "source_dataset": g.source_dataset,
                "domain": g.domain,
                "variants": variants,
                "meta": g.meta,
            }
            out_files[split].write(json.dumps(out, ensure_ascii=False) + "\n")
            counts[g.style_axis]["kept"] += 1

    for g in groups:
        variants = _dedup_variants(g.variants)
        if len(variants) < 2:
            counts[g.style_axis]["too_short"] += 1
            continue
        if apply_number_filter:
            anchor = variants[0]
            if not _passes_number_filter(anchor, variants[1:]):
                counts[g.style_axis]["number_dropped"] += 1
                continue
        if paradetox_threat_regex_filter and g.style_axis == "detox_tone":
            anchor = variants[0]
            if _paradetox_has_explicit_threat(anchor):
                counts[g.style_axis]["threat_regex_dropped"] += 1
                continue
        buffer.append(dataclasses.replace(g, variants=variants))
        if len(buffer) >= embed_batch_groups:
            flush(buffer)
            buffer = []
    flush(buffer)


def main() -> None:
    args = parse_args()
    specs = _read_config(args.config)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "train": out_dir / "style_groups_train.jsonl",
        "val": out_dir / "style_groups_val.jsonl",
        "test": out_dir / "style_groups_test.jsonl",
    }
    summary_path = out_dir / "summary.json"

    embedder = None
    if bool(args.apply_embed_filter):
        embedder = TextEmbedder(
            model_id=str(args.embed_model_id),
            cache_dir=Path(args.cache_dir),
            device=str(args.embed_device),
        )

    paradetox_risk_model = None
    if bool(args.paradetox_risk_filter):
        # Only initialize the classifier if the config actually includes detox_tone.
        if any(s.style_axis == "detox_tone" for s in specs):
            paradetox_risk_model = ParaDetoxRiskModel(
                model_id=str(args.paradetox_risk_model_id),
                cache_dir=Path(args.cache_dir),
                device=str(args.paradetox_risk_device),
            )

    max_groups_per_axis = None
    if args.max_groups_per_axis is not None and int(args.max_groups_per_axis) > 0:
        max_groups_per_axis = int(args.max_groups_per_axis)

    seen_ids: set[str] = set()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_source: Counter[str] = Counter()
    spec_errors: list[dict[str, Any]] = []

    with out_paths["train"].open("w", encoding="utf-8") as f_train, out_paths["val"].open(
        "w", encoding="utf-8"
    ) as f_val, out_paths["test"].open("w", encoding="utf-8") as f_test:
        out_files = {"train": f_train, "val": f_val, "test": f_test}
        for spec in specs:
            by_source[spec.source_dataset] += 1
            # Ensure axes appear in summary even if no examples are found.
            _ = counts[str(spec.style_axis)]
            try:
                groups = _iter_groups_for_spec(
                    spec,
                    seed=int(args.seed),
                    cache_dir=Path(args.cache_dir),
                    allow_trust_remote_code=bool(args.allow_trust_remote_code),
                )
                if spec.max_groups is not None and int(spec.max_groups) > 0:
                    groups = _take_n(groups, int(spec.max_groups))
                _filter_and_write(
                    groups,
                    embedder=embedder,
                    paradetox_risk_model=paradetox_risk_model,
                    paradetox_risk_batch_size=int(args.paradetox_risk_batch_size),
                    paradetox_threat_threshold=float(args.paradetox_threat_threshold),
                    paradetox_identity_threshold=float(args.paradetox_identity_threshold),
                    apply_number_filter=bool(args.apply_number_filter),
                    apply_embed_filter=bool(args.apply_embed_filter),
                    sim_threshold=float(spec.sim_threshold)
                    if spec.sim_threshold is not None
                    else float(args.sim_threshold),
                    embed_batch_groups=int(args.embed_batch_groups),
                    embed_text_batch_size=int(args.embed_text_batch_size),
                    paradetox_threat_regex_filter=bool(args.paradetox_threat_regex_filter),
                    seed=int(args.seed),
                    train_frac=float(args.train_frac),
                    val_frac=float(args.val_frac),
                    out_files=out_files,
                    seen_ids=seen_ids,
                    max_groups_per_axis=max_groups_per_axis,
                    counts=counts,
                )
            except Exception as exc:  # pragma: no cover - external datasets can fail in many ways
                msg = f"{spec.source_dataset} loader={spec.loader} failed: {exc}"
                print(f"[warn] {msg}")
                spec_errors.append(
                    {
                        "source_dataset": spec.source_dataset,
                        "dataset_id": spec.dataset_id,
                        "loader": spec.loader,
                        "style_axis": spec.style_axis,
                        "error": str(exc),
                    }
                )
                continue

    summary = {
        "config": str(args.config),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "apply_number_filter": bool(args.apply_number_filter),
        "apply_embed_filter": bool(args.apply_embed_filter),
        "embed_model_id": str(args.embed_model_id),
        "embed_device": str(args.embed_device),
        "sim_threshold": float(args.sim_threshold),
        "paradetox_threat_regex_filter": bool(args.paradetox_threat_regex_filter),
        "paradetox_risk_filter": bool(args.paradetox_risk_filter),
        "paradetox_risk_model_id": str(args.paradetox_risk_model_id),
        "paradetox_risk_device": str(args.paradetox_risk_device),
        "paradetox_risk_batch_size": int(args.paradetox_risk_batch_size),
        "paradetox_threat_threshold": float(args.paradetox_threat_threshold),
        "paradetox_identity_threshold": float(args.paradetox_identity_threshold),
        "sources": dict(by_source),
        "axes": {axis: dict(c) for axis, c in counts.items()},
        "spec_errors": spec_errors,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote style groups to: {out_paths}")
    print(f"Wrote summary to: {summary_path}")


def _take_n(it: Iterable[GroupCandidate], n: int) -> Iterator[GroupCandidate]:
    if n <= 0:
        return
    for i, x in enumerate(it):
        if i >= n:
            break
        yield x


if __name__ == "__main__":
    main()
