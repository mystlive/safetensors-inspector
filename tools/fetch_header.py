# -*- coding: utf-8 -*-
"""
Fetch only the header of a safetensors file hosted on Hugging Face, over HTTP Range.

The weights are never downloaded. Two requests are made: the first 8 bytes to
learn the header length, then exactly that many bytes of JSON. A 20 GB model
costs a few hundred KB.

The result can be saved as JSON for recording key names, shapes and dtypes.
No weight values are ever retrieved.

Set HF_TOKEN to reach a gated repository whose licence you have accepted.

    python tools/fetch_header.py <repo_id> <path/in/repo.safetensors> [out.json]
"""
from __future__ import annotations

import json
import os
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

UA = "safetensors-inspector/1.0 (header-only fetch; +https://github.com/mystlive/safetensors-inspector)"
MAX_HEADER = 200 * 1024 * 1024


def _request(url: str, start: int, end: int):
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end}")
    req.add_header("User-Agent", UA)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=60)


def fetch_header(repo_id: str, filename: str, revision: str = "main"):
    """(header_dict, 転送バイト数, 実際に使った URL) を返す。"""
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"

    # 1 段目: 先頭 8 バイトでヘッダ長を知る
    with _request(url, 0, 7) as r:
        status = r.status
        first = r.read(8)
    if len(first) < 8:
        raise RuntimeError(f"先頭 8 バイトを取得できない (status={status})")
    if status != 206:
        raise RuntimeError(
            f"Range リクエストが効いていない (status={status})。"
            "サーバが部分取得に対応していない可能性がある"
        )
    (n,) = struct.unpack("<Q", first)
    if n == 0 or n > MAX_HEADER:
        raise RuntimeError(f"ヘッダ長が異常: {n}")

    # 2 段目: ヘッダ本体だけを取る
    with _request(url, 8, 8 + n - 1) as r:
        raw = r.read(n)
    if len(raw) < n:
        raise RuntimeError(f"ヘッダが取り切れない: {len(raw)} < {n}")

    header = json.loads(raw.decode("utf-8"))
    return header, 8 + n, url


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    repo_id, filename = argv[0], argv[1]
    out = argv[2] if len(argv) > 2 else None

    try:
        header, nbytes, url = fetch_header(repo_id, filename)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            print(f"[{e.code}] アクセスが許可されていない: {repo_id}/{filename}")
            print("      gated リポジトリの可能性がある。"
                  "HF でライセンスに同意し、環境変数 HF_TOKEN を設定すること")
        else:
            print(f"[{e.code}] {e.reason}: {repo_id}/{filename}")
        return 2
    except Exception as e:
        print(f"[error] {repo_id}/{filename}: {e}")
        return 2

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    n_params = 0
    for v in tensors.values():
        if isinstance(v, dict):
            c = 1
            for d in v.get("shape") or []:
                c *= d
            n_params += c
    print(f"OK  {repo_id}/{filename}")
    print(f"    転送 {nbytes/1024:.1f} KB / tensors {len(tensors)} / params {n_params/1e6:.1f}M")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(header, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"    保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
