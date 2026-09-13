# Changelog

Versions are dates. There is no numbering scheme to read into.

日本語版: [CHANGELOG.ja.md](CHANGELOG.ja.md)

## 2026-09-13

- `--thumbnails DIR` writes out the preview images some files carry in their
  metadata (`modelspec.thumbnail`). The scanned tree is mirrored; scanning more
  than one target gives each its own subfolder; nothing is written outside the
  directory you name.
- The terminal now says what such a value is — `preview: JPEG image, 9.8 KB` —
  instead of counting its characters. `--meta` prints the data URI's opening
  rather than the whole base64 wall.
- The declared type is not trusted: the first bytes decide. A PNG labelled
  `image/jpeg` is written as `.png`, and a value that is not an image at all is
  reported with its declared type (`not an image: video/mp4`) rather than
  written or silently skipped.
- The HTML report runs the same check, so a value it cannot draw arrives as an
  explanation instead of a broken `<img>`.
- The same reading now reaches every output: a `preview` column in the CSV, a
  `previews` field in the JSON, and a checkbox in the GUI that puts the images
  in a folder beside the report. One function decides what counts as an image,
  so the four cannot drift apart.
- Clicking a preview in the HTML report opens it at full size.
- IP-Adapter, T2I-Adapter and CLIP Vision are identified instead of landing in
  the unidentified pile, each with the ComfyUI folder and node that reads it.
- **A CLIP published whole is no longer called a text encoder.** It carries both
  halves, and ComfyUI wants it in models/clip_vision, not models/text_encoders —
  the old answer sent it where nothing looks for it. A file holding only the
  image half is named as such.
- open_clip's text half was invisible to the component rules: its layers sit
  under `transformer.`, which the key normaliser strips as a DiT prefix, leaving
  nothing recognisable. Two keys that survive stripping now stand in for it.
- Six auxiliary models joined the verification set, which is now 53.

Measured while adding this: of the 47 public models this project verifies
against, 7 carry a preview image, all of them checkpoints or backbones from
Stability AI or Black Forest Labs. No video model carries one.

## 2026-09-07

- A banner at the top of both READMEs.

## 2026-09-05

- `stgui.py`, a small window for starting a scan, with an **Open report** button
  that reopens the report an earlier run left behind.
- A self-contained HTML report: a sortable, searchable table whose rows open
  into the full finding, with the folder summary and the unresolved detail
  folded in.
- Metadata keys are explained where reading them plainly misleads — a title that
  names the base model rather than the file, a licence that belongs to something
  else.
- A folder scan ends with a summary, and lists what it could not identify along
  with the top-level key names a rule would need.
- Merge history is read wherever the record happens to sit.
- Reports get a default name from the folder they describe.

## 2026-09-04

- First public commit. Detection rules for SD1.x, SDXL, FLUX, SD3.5,
  Qwen-Image, Wan, HunyuanVideo, HunyuanDiT, Stable Cascade, LTX-Video,
  CogVideoX, Mochi 1, SANA, PixArt, AuraFlow, Lumina 2.0, Chroma, HiDream-I1
  and Z-Image, each verified against a real file's header rather than memory.
- `tools/verify_rules.py` checks the rules against public models over HTTP
  Range, fetching headers only.
- `tools/self_check.py` checks the project against itself: counts quoted in the
  docs, rules that match nothing, messages missing from one language.
