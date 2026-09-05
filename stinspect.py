#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stinspect - tell what a safetensors file is without loading the weights.

A safetensors file starts with a JSON header:

    [8 bytes: header length N, unsigned little-endian 64-bit]
    [N bytes: UTF-8 JSON]
    [tensor data]

This tool reads that header (and a few bytes for scalar alpha values), nothing
more. Even a 20 GB checkpoint costs a few hundred KB to inspect.

Source: https://github.com/huggingface/safetensors - the Format section
  "8 bytes: N, an unsigned little-endian 64-bit integer, containing the size of the header"
  "A special key __metadata__ is allowed to contain free form string-to-string map"
Dtype names and bit widths: safetensors/src/tensor.rs, the Dtype enum and bitsize().

No third-party dependencies; standard library only.

    python stinspect.py path/to/model.safetensors
    python stinspect.py path/to/models -r --lang ja
    python stinspect.py path/to/models -r -o report.txt
    python stinspect.py path/to/models -r --csv summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import report_html
import rules
from i18n import L, tr

# safetensors/src/tensor.rs :: Dtype::bitsize()
DTYPE_BITS = {
    "BOOL": 8, "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6, "U8": 8, "I8": 8,
    "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8, "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "C64": 64, "F64": 64, "I64": 64, "U64": 64,
}

# A header longer than this is treated as a corrupt or non-safetensors file.
MAX_HEADER_BYTES = 200 * 1024 * 1024

# Below this score an architecture is a candidate, not a conclusion. Generic key
# names (blocks_N_self_attn_q_proj and the like) match unrelated models, and
# presenting a weak hit as a finding misleads more than it helps.
MIN_ARCH_SCORE = 4

# Suffixes stripped from the end of a key to get the module skeleton (longest first).
TENSOR_SUFFIXES = sorted([
    ".lora_down.weight", ".lora_up.weight", ".alpha",
    ".lora_A.weight", ".lora_B.weight",
    ".lora_magnitude_vector",
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b", ".hada_t1", ".hada_t2",
    ".lokr_w1", ".lokr_w1_a", ".lokr_w1_b",
    ".lokr_w2", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2",
    ".diff", ".diff_b", ".oft_blocks", ".oft_diag",
    ".a1.weight", ".a2.weight", ".b1.weight", ".b2.weight",
    ".on_input", ".ia3_weight",
    ".scale_weight", ".scale_input",
    ".weight", ".bias", ".gamma", ".beta",
], key=len, reverse=True)

STRIP_PREFIXES_SORTED = sorted(rules.STRIP_PREFIXES, key=lambda x: len(x[0]), reverse=True)

# Suffixes to look at when a rule needs a representative tensor for a module.
_REPRESENTATIVE = (".weight", "", ".lora_down.weight", ".lora_A.weight")


# ---------------------------------------------------------------------------
# Reading the header
# ---------------------------------------------------------------------------
class HeaderError(Exception):
    def __init__(self, key, **kw):
        self.key = key
        self.kw = kw
        super().__init__(key)

    def message(self, lang):
        return L(lang, self.key, **self.kw)


def read_header(path: Path):
    """Return (header_dict, header_length). Tensor data is never read."""
    with open(path, "rb") as f:
        raw_n = f.read(8)
        if len(raw_n) < 8:
            raise HeaderError("err_short")
        (n,) = struct.unpack("<Q", raw_n)
        if n == 0:
            raise HeaderError("err_zero")
        if n > MAX_HEADER_BYTES:
            raise HeaderError("err_huge", n=n)
        raw = f.read(n)
        if len(raw) < n:
            raise HeaderError("err_truncated")
    try:
        header = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HeaderError("err_json", err=e)
    if not isinstance(header, dict):
        raise HeaderError("err_notdict")
    return header, n


def read_scalar(path: Path, header_len: int, entry: dict):
    """Read one scalar tensor (an alpha). A seek and a few bytes, no bulk read."""
    dtype = entry.get("dtype")
    off = entry.get("data_offsets")
    if not off or dtype not in ("F16", "BF16", "F32", "F64"):
        return None
    base = 8 + header_len + off[0]
    nbytes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}[dtype]
    try:
        with open(path, "rb") as f:
            f.seek(base)
            b = f.read(nbytes)
        if len(b) < nbytes:
            return None
        if dtype == "F16":
            return struct.unpack("<e", b)[0]
        if dtype == "BF16":
            # BF16 is the top 16 bits of an F32; pad the low half with zeros.
            return struct.unpack("<f", b"\x00\x00" + b)[0]
        if dtype == "F32":
            return struct.unpack("<f", b)[0]
        return struct.unpack("<d", b)[0]
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------
def split_key(key: str):
    """Split a key into (skeleton, suffix, component).

    The skeleton has the prefix and the tensor suffix removed and separators
    unified to "_", so kohya's
        lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight
    and the diffusers form
        unet.down_blocks.0.resnets.0.conv1.lora_down.weight
    both reduce to
        down_blocks_0_resnets_0_conv1
    """
    component = None
    body = key
    for pfx, comp in STRIP_PREFIXES_SORTED:
        if body.startswith(pfx):
            body = body[len(pfx):]
            component = comp
            break

    suffix = ""
    for sfx in TENSOR_SUFFIXES:
        if body.endswith(sfx):
            suffix = sfx
            body = body[: -len(sfx)]
            break

    return body.replace(".", "_"), suffix, component


def build_modules(header: dict):
    """Return ({skeleton: {suffix: entry}}, Counter of prefix-implied components)."""
    modules = defaultdict(dict)
    prefix_components = Counter()
    for key, entry in header.items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        skel, sfx, comp = split_key(key)
        modules[skel][sfx] = entry
        if comp:
            prefix_components[comp] += 1
    return modules, prefix_components


def _representatives(modules, pattern, limit=8):
    """Every tensor entry whose skeleton matches `pattern`, up to `limit`.

    Returning only the first match makes the answer depend on dictionary order,
    which silently broke the HunyuanVideo VAE once already: two tensors carried
    the latent width on different axes and whichever was reached first won.
    """
    p = re.compile(pattern)
    found = []
    for skel, entries in modules.items():
        if not p.search(skel):
            continue
        for sfx in _REPRESENTATIVE:
            e = entries.get(sfx)
            if e and e.get("shape"):
                found.append(e)
                break
        if len(found) >= limit:
            break
    return found


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_adapter(header: dict):
    """Identify the adapter dialect. Returns (dialect, weighted hits) or (None, 0).

    Suffixes alone confuse the kohya layout with OneTrainer's internal one, so
    rules that pin down the prefix as well are preferred.
    """
    keys = [k for k in header if k != "__metadata__"]
    best, best_score = None, 0
    for d in rules.ADAPTER_DIALECTS:
        pats = [re.compile(p) for p in d["patterns"]]
        pre = d.get("prefix_pattern")
        pre_re = re.compile(pre) if pre else None
        hits = 0
        for k in keys:
            if not any(p.search(k) for p in pats):
                continue
            if pre_re and not pre_re.match(k):
                continue
            hits += 1
        score = hits * (2 if pre_re else 1)
        if hits and score > best_score:
            best, best_score = d, score
    return best, best_score


def detect_components(modules: dict, prefix_components: Counter):
    skels = list(modules)
    found = []
    for rule in rules.COMPONENT_RULES:
        pats = [re.compile(p) for p in rule["patterns"]]
        hits = sum(1 for s in skels if any(p.search(s) for p in pats))
        if hits:
            found.append({"id": rule["id"], "name": rule["name"],
                          "hits": hits, "verified": rule["verified"]})
    # Components only visible from the prefix (e.g. "first_stage_model.").
    by_id = {r["id"]: r for r in rules.COMPONENT_RULES}
    if prefix_components.get("vae") and not any(f["id"].startswith("vae") for f in found):
        found.append({"id": "vae_sd", "name": by_id["vae_sd"]["name"],
                      "hits": prefix_components["vae"], "verified": "measured"})
    if prefix_components.get("text_encoder") and not any(
        f["id"].startswith("text_encoder") for f in found
    ):
        found.append({"id": "text_encoder_generic",
                      "name": by_id["text_encoder_generic"]["name"],
                      "hits": prefix_components["text_encoder"], "verified": "measured"})
    found.sort(key=lambda x: -x["hits"])
    return found


def detect_context_dim(modules: dict):
    """Measure the cross-attention input width - the sturdiest base-model clue."""
    pats = [re.compile(p) for p in rules.CROSS_ATTN_PATTERNS]
    dims = Counter()
    for skel, entries in modules.items():
        if not any(p.search(skel) for p in pats):
            continue
        for sfx in (".lora_down.weight", ".lora_A.weight", ".weight", ""):
            e = entries.get(sfx)
            if e and e.get("shape"):
                if len(e["shape"]) >= 2:
                    dims[e["shape"][-1]] += 1
                break
    return dims


def score_architectures(modules: dict, ctx_dims: Counter, metadata: dict, kind: str):
    """Rank architecture candidates.

    `kind` narrows the field: VAE rules make no sense for a checkpoint, and a
    checkpoint bundles a CLIP encoder that would otherwise trip text-encoder rules.
    """
    if kind == "vae":
        allowed = {"vae"}
    elif kind == "text_encoder":
        allowed = {"text_encoder"}
    elif kind == "embedding":
        allowed = {"embedding"}
    elif kind == "unknown":
        allowed = {"diffusion", "vae", "text_encoder", "embedding"}
    else:
        allowed = {"diffusion"}

    skels = list(modules)
    results = []
    for arch in rules.ARCHITECTURES:
        if not (set(arch.get("for", ["diffusion"])) & allowed):
            continue

        if any(re.search(v, s) for v in arch.get("veto", []) for s in skels):
            continue

        # If such a tensor exists, its rank must match. Absent tensor: skip the
        # check, since another dialect may name it differently.
        # A required condition is satisfied if ANY matching tensor satisfies it.
        # If nothing matches the pattern at all the condition does not apply -
        # another dialect may name that tensor differently.
        dropped = False
        for pattern, want_ndim in arch.get("require_ndim", []):
            cands = _representatives(modules, pattern)
            if cands and not any(len(e["shape"]) == want_ndim for e in cands):
                dropped = True
                break
        # Same idea for a specific axis, used where two variants share every key
        # name and differ only in a width (e.g. SD1.x vs SD2.x embeddings).
        for pattern, axis, want in arch.get("require_dim", []):
            cands = _representatives(modules, pattern)
            if cands and not any(len(e["shape"]) > abs(axis) and e["shape"][axis] == want
                                 for e in cands):
                dropped = True
                break
        if dropped:
            continue

        score = 0
        evidence = []
        for pattern, weight, why in arch["signals"]:
            p = re.compile(pattern)
            hits = sum(1 for s in skels if p.search(s))
            if hits:
                score += weight
                evidence.append((why, hits))

        for dim in arch.get("context_dims", []):
            if ctx_dims.get(dim):
                # The cross-attention width is the text encoder's output width.
                # A mismatched pair cannot be wired together at all, so this is
                # decisive on its own.
                score += 8
                evidence.append(({"en": f"cross-attention input width = {dim}",
                                  "ja": f"cross-attention の入力次元 = {dim}"}, None))

        for pattern, axis, expected in arch.get("hidden", []):
            cands = _representatives(modules, pattern)
            if any(len(e["shape"]) > abs(axis) and e["shape"][axis] == expected
                   for e in cands):
                score += 3
                evidence.append(({"en": f"a representative tensor is {expected} wide, as expected",
                                  "ja": f"代表テンソルの次元 {expected} が一致"}, None))

        if score:
            results.append({"arch": arch, "score": score, "evidence": evidence})

    # Cross-check against what the producing tool declared.
    declared = metadata.get("modelspec.architecture") or metadata.get("ss_base_model_version") or ""
    if declared:
        low = str(declared).lower()
        is_xl = "xl" in low
        for r in results:
            aid = r["arch"]["id"]
            # "stable-diffusion-xl-v1-base" contains "v1", so SD1.x must not
            # claim it. Check for XL first and exclude it explicitly.
            if (aid == "sdxl" and is_xl) or \
               (aid == "sd15" and not is_xl and ("v1" in low or low.startswith("sd1"))) or \
               (aid == "qwen_image" and "qwen" in low) or \
               (aid == "flux" and "flux" in low) or \
               (aid == "wan" and "wan" in low) or \
               (aid == "sd3" and "v3" in low):
                r["score"] += 4
                r["evidence"].append(({"en": f"matches the declared metadata: {declared}",
                                       "ja": f"メタデータの宣言と一致: {declared}"}, None))

    results.sort(key=lambda r: -r["score"])
    return results


def analyze_rank(modules: dict, dialect, path: Path, header_len: int):
    if not dialect:
        return None
    down_sfx = dialect.get("down_pattern")
    ranks, alphas = Counter(), Counter()
    for entries in modules.values():
        for sfx, e in entries.items():
            if down_sfx and re.search(down_sfx, sfx) and e.get("shape"):
                ranks[e["shape"][0]] += 1
            if dialect.get("alpha_pattern") and re.search(dialect["alpha_pattern"], sfx):
                v = read_scalar(path, header_len, e)
                if v is not None:
                    alphas[round(float(v), 4)] += 1
    if not ranks and not alphas:
        return None
    return {"ranks": ranks, "alphas": alphas}


def detect_quant(tensors: dict, dtypes: Counter):
    """First matching quantisation rule wins, so order them specific-first."""
    keys = list(tensors)
    for rule in rules.QUANT_RULES:
        pats = [re.compile(p) for p in rule.get("key_patterns", [])]
        if pats and all(any(p.search(k) for k in keys) for p in pats):
            return rule
        want = rule.get("dtypes") or []
        if want and any(dt in want for dt in dtypes):
            return rule
    return None


def detect_naming_mix(modules: dict):
    """Both LDM and diffusers naming in one file - seen in the wild, not a defect."""
    ldm = re.compile(r"^(input_blocks|output_blocks|middle_block)_\d+")
    dif = re.compile(r"^(down_blocks|up_blocks|mid_block)[_.]")
    return any(ldm.search(s) for s in modules) and any(dif.search(s) for s in modules)


def classify_file_kind(dialect, components, modules):
    if dialect:
        # Most dialects are LoRA-shaped deltas, but some are not: an embedding
        # or a control model declares its own kind.
        return dialect.get("kind", "lora")
    ids = {c["id"] for c in components}
    if "controlnet" in ids:
        return "controlnet"
    has_backbone = bool(ids & {"unet_ldm", "unet_diffusers", "dit_blocks"})
    has_te = bool(ids & {"text_encoder_clip", "text_encoder_llm", "text_encoder_generic"})
    has_vae = bool(ids & {"vae_sd", "vae_3d"})
    if has_backbone and has_te and has_vae:
        return "checkpoint"
    # SD3.5 and Flux are commonly published with the VAE bundled but the text
    # encoders left out, since those are large and shared between models.
    if has_backbone and has_vae:
        return "backbone_vae"
    if has_backbone:
        return "unet_only"
    if has_te:
        return "text_encoder"
    if has_vae:
        return "vae"
    return "unknown"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(path: Path):
    result = {"path": str(path), "name": path.name,
              "size_bytes": path.stat().st_size, "error": None}

    try:
        header, header_len = read_header(path)
    except HeaderError as e:
        result["error"] = e
        return result
    except OSError as e:
        result["error"] = str(e)
        return result

    metadata = header.get("__metadata__") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    tensors = {k: v for k, v in header.items()
               if k != "__metadata__" and isinstance(v, dict)}
    result["n_tensors"] = len(tensors)

    n_params = 0
    dtypes = Counter()
    end_max = 0
    for e in tensors.values():
        cnt = 1
        for d in e.get("shape") or []:
            cnt *= d
        n_params += cnt
        dtypes[e.get("dtype", "?")] += 1
        off = e.get("data_offsets")
        if off and len(off) == 2:
            end_max = max(end_max, off[1])
    result["n_params"] = n_params
    result["dtypes"] = dict(dtypes)
    result["size_expected"] = 8 + header_len + end_max
    result["size_ok"] = (result["size_expected"] == result["size_bytes"])

    modules, prefix_components = build_modules(header)
    dialect, _ = detect_adapter(header)
    components = detect_components(modules, prefix_components)
    ctx_dims = detect_context_dim(modules)
    kind = classify_file_kind(dialect, components, modules)
    arch_results = score_architectures(modules, ctx_dims, metadata, kind)
    rank_info = analyze_rank(modules, dialect, path, header_len)

    result["quant"] = detect_quant(tensors, dtypes)

    ctx_base = None
    if ctx_dims:
        dim = ctx_dims.most_common(1)[0][0]
        if dim in rules.CONTEXT_DIM_TO_BASE:
            name, ver = rules.CONTEXT_DIM_TO_BASE[dim]
            ctx_base = {"dim": dim, "name": name, "verified": ver}

    targets = []
    if dialect:
        if prefix_components.get("unet"):
            targets.append("target_unet")
        if any(prefix_components.get(c) for c in
               ("text_encoder", "text_encoder_1", "text_encoder_2")):
            targets.append("target_te")
        if prefix_components.get("vae"):
            targets.append("target_vae")

    result.update(
        metadata=metadata,
        dialect=({"id": dialect["id"], "name": dialect["name"],
                  "verified": dialect["verified"], "note": dialect["note"]}
                 if dialect else None),
        components=components,
        context_dims=dict(ctx_dims),
        context_base=ctx_base,
        lora_targets=targets,
        architectures=[
            {"id": r["arch"]["id"], "name": r["arch"]["name"], "score": r["score"],
             "verified": r["arch"]["verified"], "note": r["arch"]["note"],
             "comfy_dir": r["arch"]["comfy_dir"], "evidence": r["evidence"]}
            for r in arch_results[:3]
        ],
        rank_info=({"ranks": dict(rank_info["ranks"]), "alphas": dict(rank_info["alphas"])}
                   if rank_info else None),
        naming_mix=detect_naming_mix(modules),
        kind=kind,
        sample_keys=sorted(tensors)[:12],
        # Kept for files that could not be identified: the top-level key names
        # are the first thing you need in order to write a rule for one.
        key_prefixes=Counter(k.split(".")[0] for k in tensors).most_common(8),
    )
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def human_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def human_count(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def verified_mark(lang, verified):
    return L(lang, f"verified_{verified}")


def confidence(lang, score):
    if score >= 8:
        return L(lang, "conf_high")
    if score >= 5:
        return L(lang, "conf_medium")
    return L(lang, "conf_low")


def wrap_note(text, lang, width=54, indent=" " * 14):
    """Wrap a long note. Japanese breaks on its own punctuation; English on spaces."""
    breaks = "。、" if lang == "ja" else " "
    out, line = [], ""
    for ch in text:
        line += ch
        if len(line) >= width and ch in breaks:
            out.append(line.rstrip() if lang != "ja" else line)
            line = ""
    if line:
        out.append(line)
    return ("\n" + indent).join(out)


def disp_width(s):
    """Display width in terminal columns; CJK characters take two."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad_to(s, width):
    return s + " " * max(1, width - disp_width(s))


def evidence_text(item, lang):
    why, hits = item
    s = tr(why, lang)
    return f"{s} ({L(lang, 'hits', n=hits)})" if hits else s


def fmt_meta_value(lang, key, value, full=False):
    s = str(value)
    if not full and (key in rules.META_BULKY or len(s) > 160):
        return L(lang, "omitted", n=len(s))
    return s


def extract_top_tags(meta, top=15):
    """Most frequent training tags - the best guess at a LoRA's trigger words."""
    raw = meta.get("ss_tag_frequency")
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    counter = Counter()
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict):
                for tag, cnt in v.items():
                    try:
                        counter[tag] += int(cnt)
                    except (TypeError, ValueError):
                        pass
    return counter.most_common(top)


def _line(text, prefix="", wrap=False):
    """One line of a report row.

    `prefix` is a label that stays on the first visual line ("Caveat: "); `wrap`
    marks prose long enough to need folding. Folding itself belongs to whoever
    renders the line - a terminal needs it, a browser does not.
    """
    return {"prefix": prefix, "text": text, "wrap": wrap}


def build_view(r, lang, full_meta=False, show_keys=False):
    """Assemble everything a report shows, before any formatting.

    Both the text report and the HTML report render this, so the two cannot
    drift apart when a field is added. Nothing here knows about column widths,
    line breaks or markup.
    """
    view = {"name": r["name"], "path": r["path"], "error": None,
            "stats": "", "head_rows": [], "size_warning": None, "rows": []}

    def add(rows, label_key, first, *rest):
        rows.append({"key": label_key, "label": L(lang, label_key),
                     "lines": [first, *rest]})

    if r["error"]:
        msg = r["error"].message(lang) if isinstance(r["error"], HeaderError) else str(r["error"])
        view["error"] = f"{L(lang, 'unreadable')} {msg}"
        return view

    dt = ", ".join(f"{k}:{v}" for k, v in sorted(r["dtypes"].items(), key=lambda x: -x[1]))
    view["stats"] = (f"{human_size(r['size_bytes'])} / {r['n_tensors']} tensors / "
                     f"{human_count(r['n_params'])} params / {dt}")
    if r.get("quant"):
        add(view["head_rows"], "label_quant",
            _line(tr(r["quant"]["name"], lang) + verified_mark(lang, r["quant"]["verified"])))
    if not r["size_ok"]:
        view["size_warning"] = L(lang, "size_mismatch",
                                 expected=r["size_expected"], actual=r["size_bytes"])

    rows = view["rows"]

    # Type
    if r["dialect"]:
        d = r["dialect"]
        lines = [_line(tr(d["name"], lang) + verified_mark(lang, d["verified"]))]
        note = tr(d["note"], lang)
        if note:
            lines.append(_line(note, wrap=True))
        add(rows, "label_type", *lines)
    else:
        add(rows, "label_type", _line(L(lang, f"kind_{r['kind']}")))

    # Base model
    strong = [a for a in r["architectures"] if a["score"] >= MIN_ARCH_SCORE]
    weak = [a for a in r["architectures"] if a["score"] < MIN_ARCH_SCORE]
    if strong:
        top = strong[0]
        lines = [_line(f"{tr(top['name'], lang)}{verified_mark(lang, top['verified'])}"
                       f"   {L(lang, 'confidence')} {confidence(lang, top['score'])}")]
        seen = set()
        for item in top["evidence"]:
            s = evidence_text(item, lang)
            if s in seen:
                continue
            seen.add(s)
            lines.append(_line(f"{L(lang, 'label_evidence')}: {s}"))
            if len(seen) >= 4:
                break
        note = tr(top["note"], lang)
        if note:
            lines.append(_line(note, prefix=f"{L(lang, 'label_caveat')}: ", wrap=True))
        add(rows, "label_base", *lines)
        if len(strong) > 1:
            add(rows, "label_alt", _line(" / ".join(
                f"{tr(a['name'], lang)} ({a['score']})" for a in strong[1:])))
    elif r["context_base"]:
        cb = r["context_base"]
        add(rows, "label_base",
            _line(f"{tr(cb['name'], lang)}{verified_mark(lang, cb['verified'])}"
                  f"   {L(lang, 'confidence')} {L(lang, 'conf_high')}"),
            _line(f"{L(lang, 'label_evidence')}: cross-attention = {cb['dim']}"))
    else:
        lines = [_line(L(lang, "base_unknown"))]
        if weak:
            lines.append(_line(f"{L(lang, 'base_weak')}: " + " / ".join(
                f"{tr(a['name'], lang)} ({a['score']}: {evidence_text(a['evidence'][0], lang)})"
                for a in weak[:2])))
        lines.append(_line(L(lang, "base_hint")))
        add(rows, "label_base", *lines)

    # Contents
    if r["components"]:
        add(rows, "label_parts", _line(" / ".join(
            f"{tr(c['name'], lang)}{verified_mark(lang, c['verified'])}"
            for c in r["components"][:5])))
    if r["lora_targets"]:
        add(rows, "label_targets", _line(" + ".join(L(lang, t) for t in r["lora_targets"])))

    # rank / alpha
    ri = r["rank_info"]
    if ri:
        parts = []
        if ri["ranks"]:
            rs = sorted(ri["ranks"].items(), key=lambda x: -x[1])
            if len(rs) == 1:
                parts.append(L(lang, "rank_single", rank=rs[0][0]))
            else:
                detail = ", ".join(f"{k} ({L(lang, 'layers', n=v)})" for k, v in rs[:4])
                parts.append(L(lang, "rank_mixed", detail=detail))
        if ri["alphas"]:
            as_ = sorted(ri["alphas"].items(), key=lambda x: -x[1])
            if len(as_) == 1:
                parts.append(L(lang, "alpha_single", alpha=f"{as_[0][0]:g}"))
            else:
                detail = ", ".join(f"{k:g} ({L(lang, 'layers', n=v)})" for k, v in as_[:4])
                parts.append(L(lang, "alpha_mixed", detail=detail))
        if parts:
            add(rows, "label_strength", _line("   ".join(parts)))

    if r["naming_mix"]:
        add(rows, "label_dialect", _line(L(lang, "naming_mix_1")),
            _line(L(lang, "naming_mix_2")))

    # Metadata
    meta = r["metadata"]
    if meta:
        shown = [(label, key, meta[key]) for key, label in rules.META_HIGHLIGHT if key in meta]
        rest = [k for k in meta if k not in {s[1] for s in shown}]
        if not shown and not full_meta:
            add(rows, "label_metadata", _line(L(lang, "thin_metadata",
                                                n=len(meta),
                                                keys=", ".join(sorted(rest)[:6]))))
        else:
            first, *others = shown or [(None, None, None)]
            if first[0] is not None:
                lines = [_line(f"{tr(first[0], lang)}: "
                               f"{fmt_meta_value(lang, first[1], first[2], full_meta)}")]
            else:
                lines = [_line("")]
            for label, key, v in others:
                lines.append(_line(f"{tr(label, lang)}: "
                                   f"{fmt_meta_value(lang, key, v, full_meta)}"))
            if full_meta:
                for k in sorted(rest):
                    lines.append(_line(f"{k}: {fmt_meta_value(lang, k, meta[k], True)}"))
            elif rest:
                lines.append(_line(L(lang, "more_items", n=len(rest),
                                     keys=", ".join(sorted(rest)[:6]),
                                     ellipsis=" ..." if len(rest) > 6 else "")))
            add(rows, "label_metadata", *lines)
        tags = extract_top_tags(meta)
        if tags:
            add(rows, "label_triggers",
                _line(", ".join(f"{t} ({c})" for t, c in tags)))
    else:
        add(rows, "label_metadata", _line(L(lang, "no_metadata")))

    # Placement
    comfy_dir, how = rules.PLACEMENT.get(r["kind"], rules.PLACEMENT["unknown"])
    if r["kind"] == "unet_only" and strong:
        comfy_dir = "models/" + strong[0]["comfy_dir"]
    lines = [_line(f"ComfyUI: {tr(comfy_dir, lang)}")]
    how = tr(how, lang)
    if how:
        lines.append(_line(how))
    add(rows, "label_placement", *lines)

    if show_keys and r["sample_keys"]:
        add(rows, "label_keys", *[_line(k) for k in r["sample_keys"]])

    return view


def print_report(r, args, out=sys.stdout):
    view = build_view(r, args.lang, full_meta=args.meta, show_keys=args.keys)
    lang = args.lang
    w = out.write
    pad = 12

    def render(line):
        prefix, text = line["prefix"], line["text"]
        if line["wrap"]:
            text = wrap_note(text, lang,
                             indent=" " * (pad + 2 + disp_width(prefix)))
        return prefix + text

    def emit(rows):
        for entry in rows:
            for i, line in enumerate(entry["lines"]):
                head = pad_to(entry["label"], pad) if i == 0 else " " * pad
                w(f"  {head}{render(line)}\n")

    w("\n" + "=" * 78 + "\n")
    w(f"{view['name']}\n")
    w(f"  {view['path']}\n")
    if view["error"]:
        w(f"  {view['error']}\n")
        return

    w(f"  {view['stats']}\n")
    emit(view["head_rows"])
    if view["size_warning"]:
        w("  " + view["size_warning"] + "\n")
    w("-" * 78 + "\n")
    emit(view["rows"])


def base_of(r, lang):
    """The base a file was identified as, or None."""
    strong = [a for a in r.get("architectures", []) if a["score"] >= MIN_ARCH_SCORE]
    if strong:
        return tr(strong[0]["name"], lang)
    if r.get("context_base"):
        return tr(r["context_base"]["name"], lang)
    return None


def unresolved_files(results):
    """Files whose base could not be established - the ones worth a new rule.

    Readable errors (a truncated download) are not in here; those are a
    different problem and are reported per file.
    """
    out = []
    for r in results:
        if r["error"]:
            continue
        if base_of(r, "en") is None:
            out.append(r)
    return out


def build_summary(results, lang):
    """The tail of a multi-file run, before formatting.

    The text report prints this; the HTML report shows the same thing in a
    panel. Wrapping stays with whoever renders it.
    """
    ok = [r for r in results if not r["error"]]
    bad = [r for r in results if r["error"]]

    groups = []
    kinds = Counter()
    for r in ok:
        kinds[tr(r["dialect"]["name"], lang) if r["dialect"]
              else L(lang, f"kind_{r['kind']}")] += 1
    if kinds:
        groups.append({"title": L(lang, "summary_by_type"),
                       "counts": kinds.most_common()})

    bases = Counter()
    for r in ok:
        b = base_of(r, lang)
        bases[b if b else L(lang, "base_unknown")] += 1
    if bases:
        groups.append({"title": L(lang, "summary_by_base"),
                       "counts": bases.most_common()})

    damaged = None
    if bad:
        damaged = {"title": L(lang, "summary_damaged", n=len(bad)),
                   "items": [{"name": r["name"], "text": error_text(r, lang)}
                             for r in bad]}

    unres = unresolved_files(ok)
    unresolved = None
    if unres:
        unresolved = {
            "title": L(lang, "summary_unresolved", n=len(unres)),
            "keys_label": L(lang, "unresolved_keys"),
            "items": [{"name": r["name"],
                       "keys": ", ".join(f"{k} ({n})"
                                         for k, n in r.get("key_prefixes", []))}
                      for r in unres],
            # Only the last hint is long enough to need folding in a terminal.
            "hints": [{"text": L(lang, "unresolved_hint_1"), "wrap": False},
                      {"text": L(lang, "unresolved_hint_2"), "wrap": False},
                      {"text": L(lang, "unresolved_hint_3"), "wrap": True}],
        }

    return {"title": L(lang, "summary_title", n=len(results)),
            "groups": groups, "damaged": damaged, "unresolved": unresolved}


def print_summary(results, args, out=sys.stdout):
    lang = args.lang
    s = build_summary(results, lang)
    w = out.write

    w("\n" + "=" * 78 + "\n")
    w(s["title"] + "\n")

    for g in s["groups"]:
        w("\n" + g["title"] + "\n")
        for name, n in g["counts"]:
            w(f"  {n:>4}  {name}\n")

    if s["damaged"]:
        w("\n" + s["damaged"]["title"] + "\n")
        for it in s["damaged"]["items"]:
            w(f"  {it['name']}: {it['text']}\n")

    if s["unresolved"]:
        u = s["unresolved"]
        w("\n" + u["title"] + "\n")
        for it in u["items"]:
            w(f"  {it['name']}\n")
            w(f"      {u['keys_label']}: {it['keys']}\n")
        w("\n")
        for h in u["hints"]:
            text = (wrap_note(h["text"], lang, width=70, indent="  ")
                    if h["wrap"] else h["text"])
            w("  " + text + "\n")


def build_unresolved(results, lang):
    """Per-file detail for the files that stumped it: what a rule would need.

    Written out by --unresolved and shown as a panel in the HTML report.
    """
    items = []
    for r in unresolved_files([r for r in results if not r["error"]]):
        dt = ", ".join(f"{k}:{v}"
                       for k, v in sorted(r["dtypes"].items(), key=lambda x: -x[1]))
        rows = [
            {"label": L(lang, "unresolved_keys"), "list": False,
             "value": ", ".join(f"{k} ({n})" for k, n in r.get("key_prefixes", [])),
             "items": []},
        ]
        comps = [tr(c["name"], lang) for c in r.get("components", [])]
        rows.append({"label": L(lang, "label_parts"), "list": False,
                     "value": " / ".join(comps) if comps else "-", "items": []})
        if r.get("metadata"):
            rows.append({"label": L(lang, "label_metadata"), "list": False,
                         "value": ", ".join(sorted(r["metadata"])[:10]), "items": []})
        weak = [a for a in r.get("architectures", []) if a["score"] < MIN_ARCH_SCORE]
        if weak:
            rows.append({"label": L(lang, "base_weak"), "list": False,
                         "value": " / ".join(f"{tr(a['name'], lang)} ({a['score']})"
                                             for a in weak[:3]),
                         "items": []})
        rows.append({"label": L(lang, "label_keys"), "list": True, "value": "",
                     "items": list(r.get("sample_keys", []))})
        items.append({
            "name": r["name"], "path": r["path"],
            "stats": (f"{human_size(r['size_bytes'])} / {r['n_tensors']} tensors / "
                      f"{human_count(r['n_params'])} params / {dt}"),
            "rows": rows,
        })
    return {"title": L(lang, "unresolved_file_title", n=len(items)),
            "intro": L(lang, "unresolved_file_intro"), "items": items}


def write_unresolved(results, path, lang):
    """Write the details needed to add rules for the files that stumped it."""
    u = build_unresolved(results, lang)
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("# " + u["title"] + "\n")
        f.write(u["intro"] + "\n\n")
        for it in u["items"]:
            f.write("=" * 78 + "\n")
            f.write(f"{it['name']}\n")
            f.write(f"  {it['path']}\n")
            f.write(f"  {it['stats']}\n")
            for row in it["rows"]:
                if row["list"]:
                    f.write(f"  {row['label']}:\n")
                    for k in row["items"]:
                        f.write(f"      {k}\n")
                else:
                    f.write(f"  {row['label']}: {row['value']}\n")
            f.write("\n")
    return len(u["items"])


def to_jsonable(r, lang):
    """Resolve bilingual fields so the JSON output is plain strings."""
    d = dict(r)
    if isinstance(d.get("error"), HeaderError):
        d["error"] = d["error"].message(lang)
    if d.get("dialect"):
        d["dialect"] = dict(d["dialect"],
                            name=tr(d["dialect"]["name"], lang),
                            note=tr(d["dialect"]["note"], lang))
    if d.get("context_base"):
        d["context_base"] = dict(d["context_base"], name=tr(d["context_base"]["name"], lang))
    d["components"] = [dict(c, name=tr(c["name"], lang)) for c in d.get("components", [])]
    d["architectures"] = [
        dict(a, name=tr(a["name"], lang), note=tr(a["note"], lang),
             evidence=[evidence_text(e, lang) for e in a["evidence"]])
        for a in d.get("architectures", [])
    ]
    d["lora_targets"] = [L(lang, t) for t in d.get("lora_targets", [])]
    if d.get("quant"):
        d["quant"] = tr(d["quant"]["name"], lang)
    return d


def summary_fields(r, lang):
    """The one-line-per-file view: type, base, confidence, rank.

    Shared by the CSV and the HTML table so a file cannot be summarised one way
    in one and another way in the other.
    """
    strong = [a for a in r["architectures"] if a["score"] >= MIN_ARCH_SCORE]
    if strong:
        base, conf = tr(strong[0]["name"], lang), confidence(lang, strong[0]["score"])
    elif r["context_base"]:
        base, conf = tr(r["context_base"]["name"], lang), L(lang, "conf_high")
    else:
        base = conf = ""
    rank = ""
    if r["rank_info"] and r["rank_info"]["ranks"]:
        rs = sorted(r["rank_info"]["ranks"].items(), key=lambda x: -x[1])
        rank = str(rs[0][0]) if len(rs) == 1 else f"{rs[0][0]}+"
    kind = tr(r["dialect"]["name"], lang) if r["dialect"] else L(lang, f"kind_{r['kind']}")
    return kind, base, conf, rank


def error_text(r, lang):
    if not r["error"]:
        return None
    return (r["error"].message(lang) if isinstance(r["error"], HeaderError)
            else str(r["error"]))


def write_csv(results, path, lang):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([L(lang, k) for k in
                     ("csv_name", "csv_kind", "csv_base", "csv_conf", "csv_rank",
                      "csv_size", "csv_tensors", "csv_params", "csv_path", "csv_error")])
        for r in results:
            if r["error"]:
                wr.writerow([r["name"], "", "", "", "", "", "", "", r["path"],
                             error_text(r, lang)])
                continue
            kind, base, conf, rank = summary_fields(r, lang)
            wr.writerow([r["name"], kind, base, conf, rank,
                         human_size(r["size_bytes"]), r["n_tensors"],
                         human_count(r["n_params"]), r["path"], ""])


# How many table rows the HTML report draws before offering "show more".
# Rendering cost tracks DOM node count: 8000 rows drawn at once measured ~2s to
# load and ~1s per keystroke in the search box, while the 18 MB JSON payload
# behind them parsed in ~110ms. Holding the table down keeps the page usable
# whatever the folder size.
HTML_ROW_CHUNK = 500

HTML_COLUMNS = (
    # key; label key (shared with the CSV so both name a column the same way);
    # numeric (sorts by value, not text); clip (one line with an ellipsis in the
    # table - type and base are explanatory sentences, and letting them wrap
    # makes every row several lines tall. The full text is in the detail panel).
    ("name", "csv_name", False, False),
    ("kind", "csv_kind", False, True),
    ("base", "csv_base", False, True),
    ("conf", "csv_conf", False, False),
    ("rank", "csv_rank", False, False),
    ("size", "csv_size", True, False),
    ("tensors", "csv_tensors", True, False),
    ("params", "csv_params", True, False),
)


def build_page(results, lang, full_meta=False, show_keys=False, summary=True):
    """Everything report_html.py needs, with every string already in `lang`.

    Details come from build_view, and the two panels from build_summary and
    build_unresolved - the same structures the text output renders.
    """
    rows = []
    for i, r in enumerate(results):
        view = build_view(r, lang, full_meta=full_meta, show_keys=show_keys)
        if r["error"]:
            kind = base = conf = rank = tensors = params = ""
            sort = {"size": r["size_bytes"], "tensors": 0, "params": 0}
        else:
            kind, base, conf, rank = summary_fields(r, lang)
            tensors = str(r["n_tensors"])
            params = human_count(r["n_params"])
            sort = {"size": r["size_bytes"], "tensors": r["n_tensors"],
                    "params": r["n_params"]}
        rows.append({
            "id": i,
            "name": r["name"],
            "path": r["path"],
            "kind": kind, "base": base, "conf": conf, "rank": rank,
            "size": human_size(r["size_bytes"]),
            "tensors": tensors, "params": params,
            "sort": sort,
            "error": view["error"] or "",
            "detail": {
                "stats": view["stats"],
                "size_warning": view["size_warning"] or "",
                # The quantisation row sits above the rule in the text report;
                # in a browser there is no rule, so it simply leads the list.
                "rows": view["head_rows"] + view["rows"],
            },
            "haystack": " ".join((r["name"], r["path"], kind, base)).lower(),
        })

    # One file says everything in its own detail, so the panels only earn their
    # place on a folder scan - the same rule the text report follows.
    panels = summary and len(results) > 1
    detail = build_unresolved(results, lang) if panels else None

    return {
        "lang": lang,
        "title": L(lang, "html_title"),
        "chunk": HTML_ROW_CHUNK,
        "summary": build_summary(results, lang) if panels else None,
        "unresolved": detail if (detail and detail["items"]) else None,
        "ui": {
            "search": L(lang, "html_search"),
            "all_kinds": L(lang, "html_all_kinds"),
            # Left unformatted on purpose: these counts change as you filter,
            # so the browser fills in the placeholders.
            "showing": L(lang, "html_showing"),
            "show_more": L(lang, "html_show_more"),
            "no_match": L(lang, "html_no_match"),
        },
        "columns": [{"key": k, "label": L(lang, lk), "numeric": num, "clip": clip}
                    for k, lk, num, clip in HTML_COLUMNS],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def collect_files(targets, recursive, lang):
    files = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.safetensors") if recursive
                                else p.glob("*.safetensors")))
        elif p.is_file():
            files.append(p)
        else:
            print(L(lang, "skip", path=t), file=sys.stderr)
    return files


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lang", choices=("en", "ja"), default="en")
    known, _ = pre.parse_known_args()
    lang = known.lang

    ap = argparse.ArgumentParser(
        description=L(lang, "help_desc"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", default=["."], help=L(lang, "help_targets"))
    ap.add_argument("-r", "--recursive", action="store_true", help=L(lang, "help_recursive"))
    ap.add_argument("--meta", action="store_true", help=L(lang, "help_meta"))
    ap.add_argument("--keys", action="store_true", help=L(lang, "help_keys"))
    ap.add_argument("--json", action="store_true", help=L(lang, "help_json"))
    ap.add_argument("--csv", metavar="PATH", help=L(lang, "help_csv"))
    ap.add_argument("--html", metavar="PATH", help=L(lang, "help_html"))
    ap.add_argument("-o", "--out", metavar="PATH", help=L(lang, "help_out"))
    ap.add_argument("--unresolved", metavar="PATH", help=L(lang, "help_unresolved"))
    ap.add_argument("--no-summary", action="store_true", help=L(lang, "help_no_summary"))
    ap.add_argument("--lang", choices=("en", "ja"), default="en", help=L(lang, "help_lang"))
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    files = collect_files(args.targets or ["."], args.recursive, args.lang)
    if not files:
        print(L(args.lang, "not_found"), file=sys.stderr)
        return 1

    results = [analyze(p) for p in files]

    # BOM-prefixed UTF-8: without it Notepad guesses the local codepage and mojibake results.
    out_fh = open(args.out, "w", encoding="utf-8-sig", newline="\n") if args.out else None
    stream = out_fh or sys.stdout
    try:
        if args.json:
            stream.write(json.dumps([to_jsonable(r, args.lang) for r in results],
                                    ensure_ascii=False, indent=2) + "\n")
        else:
            for r in results:
                print_report(r, args, out=stream)
            # A single file already says everything in its own report.
            if len(results) > 1 and not args.no_summary:
                print_summary(results, args, out=stream)
            else:
                stream.write("\n" + "=" * 78 + "\n")
                stream.write(L(args.lang, "summary", n=len(results)) + "\n")
    finally:
        if out_fh:
            out_fh.close()
            print(L(args.lang, "wrote", path=args.out))

    if args.csv:
        write_csv(results, args.csv, args.lang)
        print(L(args.lang, "wrote_csv", path=args.csv))

    if args.html:
        report_html.write_html(
            build_page(results, args.lang, full_meta=args.meta,
                       show_keys=args.keys, summary=not args.no_summary),
            args.html)
        print(L(args.lang, "wrote_html", path=args.html))

    if args.unresolved:
        n = write_unresolved(results, args.unresolved, args.lang)
        print(L(args.lang, "wrote_unresolved", path=args.unresolved, n=n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
