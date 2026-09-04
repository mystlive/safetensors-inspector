# -*- coding: utf-8 -*-
"""Dump the raw key structure of a safetensors file.

Use this when stinspect cannot identify a file and you want to write a rule for
it: it prints the metadata, dtype spread, key prefixes grouped by depth, and a
sample of keys with their shapes.

    python tools/probe_header.py path/to/file.safetensors
"""
import json
import struct
import sys
from collections import Counter
from pathlib import Path

MAX_HEADER = 200 * 1024 * 1024


def read_header(path):
    with open(path, "rb") as f:
        n_raw = f.read(8)
        if len(n_raw) < 8:
            raise ValueError("too short")
        (n,) = struct.unpack("<Q", n_raw)
        if n == 0 or n > MAX_HEADER:
            raise ValueError(f"header size implausible: {n}")
        raw = f.read(n)
        if len(raw) < n:
            raise ValueError("truncated header")
    return json.loads(raw.decode("utf-8"))


def prefix(key, depth):
    parts = key.split(".")
    return ".".join(parts[:depth])


def main(paths):
    for p in paths:
        p = Path(p)
        print("=" * 100)
        print(p)
        try:
            h = read_header(p)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        meta = h.pop("__metadata__", None)
        print(f"  tensors: {len(h)}  file_size: {p.stat().st_size/1048576:.1f} MB")
        if meta:
            print("  __metadata__:")
            for k, v in meta.items():
                vs = str(v)
                if len(vs) > 200:
                    vs = vs[:200] + f"...(len={len(str(v))})"
                print(f"    {k} = {vs}")
        else:
            print("  __metadata__: (none)")

        dt = Counter(v.get("dtype") for v in h.values())
        print(f"  dtypes: {dict(dt)}")

        for d in (1, 2, 3):
            c = Counter(prefix(k, d) for k in h)
            top = c.most_common(14)
            print(f"  prefix depth{d} ({len(c)} uniq): {top}")

        keys = sorted(h)
        print("  sample keys:")
        for k in keys[:8]:
            print(f"    {k}  {h[k].get('shape')}  {h[k].get('dtype')}")
        if len(keys) > 16:
            mid = len(keys) // 2
            for k in keys[mid:mid + 4]:
                print(f"    {k}  {h[k].get('shape')}  {h[k].get('dtype')}")
        for k in keys[-4:]:
            print(f"    {k}  {h[k].get('shape')}  {h[k].get('dtype')}")

        markers = [
            "lora_down", "lora_up", "lora_A", "lora_B", ".alpha",
            "hada_w1", "lokr_", "oft_", "diff_b",
            "model.diffusion_model", "cond_stage_model", "conditioner.embedders",
            "first_stage_model", "double_blocks", "single_blocks",
            "transformer_blocks", "input_blocks", "output_blocks", "middle_block",
            "time_embed", "label_emb", "text_model", "encoder.block",
            "control_model", "controlnet", "img_in", "txt_in",
            "decoder.up_blocks", "encoder.down_blocks", "quant_conv",
            "emb_params", "string_to_param", "logit_scale",
        ]
        hit = [m for m in markers if any(m in k for k in h)]
        print(f"  markers: {hit}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
