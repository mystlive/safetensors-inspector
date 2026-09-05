# -*- coding: utf-8 -*-
"""
Check the project against itself: do the docs' claims match the code, and do
the rules hold together?

This is not a test of whether files are identified correctly - tools/verify_rules.py
does that. This checks the things that rot quietly:

  - counts quoted in the READMEs and the key reference vs the actual rule tables
  - every verification target appearing in the documented model list
  - message keys used by the code existing in both languages, including the
    launcher window's GUI_KEYS list matching what it actually asks for
  - the two language catalogues holding the same set of keys
  - every rule carrying both languages, a unique id, and valid regexes
  - signals that never match anything in the cached headers
  - the analyser surviving malformed files
  - internal documentation links resolving

Run it after changing rules or docs:

    python tools/self_check.py

The cached-header checks need tools/headers/, which tools/verify_rules.py fills
in. Without it those checks are skipped rather than failed.
"""
from __future__ import annotations

import json
import re
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import report_html  # noqa: E402
import rules       # noqa: E402
import stinspect   # noqa: E402
import verify_rules  # noqa: E402
from i18n import LABELS  # noqa: E402

HDR = ROOT / "tools" / "headers"
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
        print(f"  FAIL  {msg}")
    else:
        print(f"  ok    {msg}")


def section(title):
    print(f"\n== {title} ==")


ALL_RULES = [(n, g) for n, g in (
    ("ARCHITECTURES", rules.ARCHITECTURES),
    ("ADAPTER_DIALECTS", rules.ADAPTER_DIALECTS),
    ("COMPONENT_RULES", rules.COMPONENT_RULES),
    ("QUANT_RULES", rules.QUANT_RULES),
)]

docs = {f: (ROOT / f).read_text(encoding="utf-8")
        for f in ("README.md", "README.ja.md", "docs/key-reference.md")}

# ---------------------------------------------------------------------------
section("rule counts vs the numbers quoted in the docs")
tally = Counter()
for _, group in ALL_RULES:
    for r in group:
        tally[r.get("verified", "?")] += 1
print(f"  actual: {dict(tally)}")

for f, txt in docs.items():
    for m in re.finditer(r"(\d+) rules measured", txt):
        check(int(m.group(1)) == tally["measured"],
              f"{f}: says {m.group(1)} measured, actual {tally['measured']}")
    for m in re.finditer(r"実測 (\d+) / 導出 (\d+)", txt):
        check(int(m.group(1)) == tally["measured"], f"{f}: measured count")
        check(int(m.group(2)) == tally["derived"], f"{f}: derived count")
    for m in re.finditer(r"tally today is (\d+) measured, (\d+) derived, (\d+) unverified", txt):
        check(int(m.group(1)) == tally["measured"], f"{f}: tally measured")
        check(int(m.group(2)) == tally["derived"], f"{f}: tally derived")
        check(int(m.group(3)) == tally.get("unverified", 0), f"{f}: tally unverified")

# ---------------------------------------------------------------------------
section("verification targets vs the numbers quoted in the docs")
n_targets = len(verify_rules.TARGETS)
n_gated = sum(1 for t in verify_rules.TARGETS if t["gated"])
print(f"  actual: {n_targets} targets, {n_gated} gated, {n_targets - n_gated} ungated")
for f, txt in docs.items():
    for m in re.finditer(r"(\d+) public models", txt):
        check(int(m.group(1)) == n_targets, f"{f}: says {m.group(1)} models")
    for m in re.finditer(r"公開モデル (\d+) 件", txt):
        check(int(m.group(1)) == n_targets, f"{f}: says {m.group(1)} models")
    for m in re.finditer(r"other (\d+) still (?:pass|run)", txt):
        check(int(m.group(1)) == n_targets - n_gated, f"{f}: says {m.group(1)} ungated")
    for m in re.finditer(r"残り (\d+) 件は通る", txt):
        check(int(m.group(1)) == n_targets - n_gated, f"{f}: says {m.group(1)} ungated")

# ---------------------------------------------------------------------------
section("every verification target is in the documented model list")
listed = set(re.findall(r"^\| `([\w\-./]+)` \| `([^`]+)` \|", docs["docs/key-reference.md"], re.M))
targets = {(t["repo"], t["file"]) for t in verify_rules.TARGETS}
check(not (targets - listed), f"targets missing from the docs: {sorted(targets - listed)[:3]}")
check(not (listed - targets), f"documented but not a target: {sorted(listed - targets)[:3]}")

# ---------------------------------------------------------------------------
section("message keys used by the code exist in both languages")
src = (ROOT / "stinspect.py").read_text(encoding="utf-8")
used = set(re.findall(r'L\((?:args\.)?lang,\s*"([a-z0-9_]+)"', src))
used |= set(re.findall(r'add\([^,]+,\s*"([a-z0-9_]+)"', src))
used |= {f"kind_{k}" for k in ("checkpoint", "unet_only", "backbone_vae", "text_encoder",
                               "vae", "controlnet", "embedding", "unknown")}
used |= {f"verified_{v}" for v in ("measured", "derived", "unverified")}
used |= {"target_unet", "target_te", "target_vae"}

# stgui.py declares its keys in GUI_KEYS. Read the list out of the source
# rather than importing the module, which would require Tk to be installed.
gui_src = (ROOT / "stgui.py").read_text(encoding="utf-8")
block = re.search(r"GUI_KEYS = \((.*?)\)", gui_src, re.S)
check(block is not None, "stgui.py declares GUI_KEYS")
if block:
    declared = set(re.findall(r'"([a-z0-9_]+)"', block.group(1)))
    body = gui_src[:block.start()] + gui_src[block.end():]
    called = set(re.findall(r'"(gui_[a-z0-9_]+)"', body))
    check(not (called - declared),
          f"gui keys used but not declared: {sorted(called - declared)[:6]}")
    check(not (declared - called),
          f"declared in GUI_KEYS but never used: {sorted(declared - called)[:6]}")
    used |= declared

for lang in ("en", "ja"):
    miss = sorted(k for k in used if k not in LABELS[lang])
    check(not miss, f"{lang}: missing keys {miss[:6]}")

section("the two language catalogues hold the same keys")
check(not set(LABELS["en"]) - set(LABELS["ja"]),
      f"only in en: {sorted(set(LABELS['en']) - set(LABELS['ja']))[:6]}")
check(not set(LABELS["ja"]) - set(LABELS["en"]),
      f"only in ja: {sorted(set(LABELS['ja']) - set(LABELS['en']))[:6]}")

# ---------------------------------------------------------------------------
section("every rule is well formed")
bare = []
for name, group in ALL_RULES:
    for r in group:
        for field in ("name", "note"):
            v = r.get(field)
            if v is None:
                continue
            if not isinstance(v, dict) or "en" not in v or "ja" not in v:
                bare.append(f"{name}/{r['id']}.{field}")
check(not bare, f"fields not wrapped in T(en, ja): {bare[:6]}")

ids = [r["id"] for _, g in ALL_RULES for r in g]
check(len(ids) == len(set(ids)),
      f"duplicate rule ids: {[k for k, n in Counter(ids).items() if n > 1]}")

bad_re = []
for name, group in ALL_RULES:
    for r in group:
        pats = list(r.get("patterns", [])) + list(r.get("veto", [])) \
             + list(r.get("key_patterns", [])) \
             + [s[0] for s in r.get("signals", [])] \
             + [p for p, _ in r.get("require_ndim", [])] \
             + [p for p, _, _ in r.get("require_dim", [])] \
             + [p for p, _, _ in r.get("hidden", [])]
        if r.get("prefix_pattern"):
            pats.append(r["prefix_pattern"])
        for p in pats:
            try:
                re.compile(p)
            except re.error as e:
                bad_re.append(f"{r['id']}: {p!r} ({e})")
check(not bad_re, f"invalid regexes: {bad_re[:3]}")

# ---------------------------------------------------------------------------
section("signals match something in the cached headers")
# Rules whose sample is not among the verification targets cannot be judged here.
# They are checked by running stinspect against a local copy instead.
NOT_IN_CORPUS = {"anima", "te_qwen_vl", "te_qwen_llm", "vae_3d_16ch", "vae_3d",
                 "text_encoder_llm", "sd3", "ti_sd2"}
headers = sorted(HDR.glob("*.json"))
if not headers:
    print("  skipped: no cached headers. Run tools/verify_rules.py first.")
else:
    corpus = set()
    for hp in headers:
        h = json.loads(hp.read_text(encoding="utf-8"))
        h.pop("__metadata__", None)
        mods, _ = stinspect.build_modules(h)
        corpus |= set(mods)
    print(f"  corpus: {len(corpus)} skeletons from {len(headers)} headers")
    dead = []
    for name, group in (("ARCHITECTURES", rules.ARCHITECTURES),
                        ("COMPONENT_RULES", rules.COMPONENT_RULES)):
        for r in group:
            if r["id"] in NOT_IN_CORPUS:
                continue
            pats = [s[0] for s in r.get("signals", [])] or list(r.get("patterns", []))
            for p in pats:
                if not any(re.search(p, s) for s in corpus):
                    dead.append(f"{r['id']}: {p}")
    check(not dead, f"signals that match nothing: {dead[:5]}")

# ---------------------------------------------------------------------------
section("malformed files are handled without crashing")
tmp = Path(tempfile.mkdtemp())


def write_st(name, header, data=b""):
    raw = json.dumps(header).encode("utf-8")
    p = tmp / name
    p.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return p


class Args:
    lang = "en"; meta = False; keys = False; no_summary = False


class Sink:
    def __init__(self):
        self.buf = []

    def write(self, s):
        self.buf.append(s)


cases = [
    ("empty header", write_st("a.safetensors", {})),
    ("metadata is not a dict", write_st("b.safetensors", {"__metadata__": "oops"})),
    ("scalar tensor", write_st("c.safetensors",
        {"x": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}, b"\0" * 4)),
    ("tensor without a shape", write_st("d.safetensors",
        {"x": {"dtype": "F32", "data_offsets": [0, 4]}}, b"\0" * 4)),
    ("unknown dtype", write_st("e.safetensors",
        {"x": {"dtype": "WAT", "shape": [2], "data_offsets": [0, 8]}}, b"\0" * 8)),
    ("entry is not a dict", write_st("f.safetensors", {"x": 42})),
    ("reversed offsets", write_st("g.safetensors",
        {"x": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}, b"\0" * 4)),
    ("eight zero bytes", (tmp / "h.safetensors")),
]
(tmp / "h.safetensors").write_bytes(b"\0" * 8)

results = []
for label, p in cases:
    try:
        r = stinspect.analyze(p)
        results.append(r)
        stinspect.print_report(r, Args(), out=Sink())
    except Exception as e:
        fails.append(f"{label}: {type(e).__name__}: {e}")
        print(f"  FAIL  {label}: {type(e).__name__}: {e}")
        continue
    print(f"  ok    {label}")

for label, fn in (("summary", lambda: stinspect.print_summary(results, Args(), out=Sink())),
                  ("unresolved report", lambda: stinspect.write_unresolved(results, tmp / "u.txt", "en")),
                  ("csv", lambda: stinspect.write_csv(results, tmp / "x.csv", "en")),
                  ("html", lambda: report_html.write_html(
                      stinspect.build_page(results, "en"), tmp / "r.html")),
                  ("json (ja)", lambda: json.dumps([stinspect.to_jsonable(r, "ja") for r in results],
                                                   ensure_ascii=False))):
    try:
        fn()
        print(f"  ok    {label} on malformed input")
    except Exception as e:
        fails.append(f"{label}: {type(e).__name__}: {e}")
        print(f"  FAIL  {label}: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
section("the HTML report escapes metadata that carries markup")
# ComfyUI writes its whole prompt and workflow JSON into __metadata__, so a
# report can legitimately contain angle brackets. A `</script>` in there would
# end the embedded payload; anything after it would be parsed as markup.
HOSTILE = "</script><img src=x onerror=alert(1)><!-- & -->"
hp = write_st("hostile.safetensors",
              {"__metadata__": {"note": HOSTILE},
               "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
              b"\0" * 4)
hr = stinspect.analyze(hp)
html = report_html.render(stinspect.build_page([hr], "en", full_meta=True))

check(html.count("</script") == 2,
      f"exactly the report's own two script elements close ({html.count('</script')} found)")
check(HOSTILE not in html, "the raw markup is nowhere in the document")
check("<img" not in html, "no img element was smuggled in")

m = re.search(r'<script type="application/json" id="stinspect-data">(.*?)</script>',
              html, re.S)
check(m is not None, "the payload element is intact")
if m:
    try:
        payload = json.loads(m.group(1))
        text = json.dumps(payload, ensure_ascii=False)
        check(HOSTILE in text, "the value survives escaping unchanged")
    except json.JSONDecodeError as e:
        check(False, f"the payload is not valid JSON: {e}")

# ---------------------------------------------------------------------------
section("internal documentation links resolve")
for f in ("README.md", "README.ja.md", "docs/key-reference.md",
          "docs/guide.md", "docs/guide.ja.md"):
    txt = (ROOT / f).read_text(encoding="utf-8")
    base = (ROOT / f).parent
    for m in re.finditer(r"\]\(([^)#:]+\.md)\)", txt):
        check((base / m.group(1)).resolve().exists(), f"{f} -> {m.group(1)}")

print("\n" + "=" * 62)
if fails:
    print(f"{len(fails)} problem(s):")
    for f_ in fails:
        print(f"  - {f_}")
    sys.exit(1)
print("no problems found")
