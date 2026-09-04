# safetensors-inspector

Work out what a `.safetensors` file actually is — a full model, a LoRA, a VAE, a
text encoder — and which base model it needs, without loading the weights.

For when you have a folder of downloads and no longer remember which is which.

日本語版: [README.ja.md](README.ja.md)

```
==============================================================================
sd_xl_offset_example-lora_1.0.safetensors
  models/loras/sd_xl_offset_example-lora_1.0.safetensors
  47.3 MB / 2364 tensors / 24.6M params / F16:2364
------------------------------------------------------------------------------
  Type        LoRA (kohya / sd-scripts layout)
              The most widely supported LoRA layout. A1111, ComfyUI
              and Forge read it as-is.
  Base        SDXL family (SDXL 1.0 / Illustrious / Pony / NoobAI / Animagine ...)   confidence high
              Evidence: transformers sitting on the deep blocks, as SDXL does (100 matches)
              Evidence: cross-attention input width = 2048
              Evidence: matches the declared metadata: stable-diffusion-xl-v1-base/lora
              Caveat: Derivatives such as Illustrious, Pony, NoobAI and Animagine
                      are structurally identical to SDXL 1.0, so they cannot
                      be told apart from the weights' shape. Distinguishing
                      them needs metadata or the filename.
  Contents    UNet (LDM / SAI naming)
  Targets     UNet / DiT
  Strength    rank 8   alpha 1
  Metadata    training base: sdxl_base_v0-9
              rank (network_dim): 8
              alpha: 1
              training images: 7412
              declared architecture: stable-diffusion-xl-v1-base/lora
              title: SDXL 1.0 Official Offset Example LoRA
              (49 more: ... use --meta to show all)
  Triggers    contrast (7412)
  Place in    ComfyUI: models/loras
              LoraLoader node. The base model has to match
```

*(metadata block trimmed for the README; the tool prints more)*

## Why it is fast

A safetensors file starts with a JSON header:

```
[8 bytes: header length N, unsigned little-endian 64-bit]
[N bytes: UTF-8 JSON]
[tensor data]
```

This tool reads that header and nothing else (plus a handful of bytes for scalar
alpha values). A 6 GB checkpoint costs a few hundred KB to identify, so scanning
a whole model folder takes seconds.

## Install

No dependencies. Python 3.8 or newer, standard library only — no PyTorch, not even
the `safetensors` package.

```bash
git clone https://github.com/mystlive/safetensors-inspector
cd safetensors-inspector
python stinspect.py --help
```

## Usage

```bash
# one file
python stinspect.py path/to/mystery.safetensors

# a whole model folder
python stinspect.py path/to/ComfyUI/models -r

# save a report (UTF-8 with BOM, so it opens cleanly in any editor)
python stinspect.py path/to/models -r -o report.txt

# a spreadsheet of everything you own
python stinspect.py path/to/models -r --csv inventory.csv

# Japanese output
python stinspect.py path/to/models --lang ja
```

| Option | Effect |
| --- | --- |
| `-r`, `--recursive` | walk folders recursively |
| `-o PATH` | write the report to a file (UTF-8 with BOM) |
| `--csv PATH` | write a summary table (UTF-8 with BOM, opens in Excel) |
| `--meta` | print all metadata instead of the highlights |
| `--keys` | also print sample key names, for investigating unidentified files |
| `--json` | emit JSON |
| `--lang {en,ja}` | output language (default `en`) |

Prefer `-o` over shell redirection: `>` writes UTF-8 without a BOM, which Notepad
and some editors then misread as the local codepage.

## What it tells you

- **Type** — full checkpoint, diffusion backbone only, LoRA, VAE, text encoder,
  ControlNet or embedding
- **Base model** — SD1.x, SDXL, Qwen-Image, Wan, Anima, and so on, with the
  evidence for the call
- **LoRA specifics** — rank, alpha, whether it touches the text encoder, and which
  key layout it uses (some layouts will not load in A1111 or ComfyUI as-is)
- **Where to put it** — the ComfyUI folder and the node that loads it
- **Trigger words** — from the training tag frequency, when the trainer recorded it
- **Damage** — truncated downloads and corrupt headers, from a size cross-check

Every finding comes with its evidence, and rules that have not been checked against
a real file are flagged `[unverified / inferred]` so you know how much to trust them.

## What it cannot tell you

Files that are structurally identical cannot be told apart from the header. The
tool says so rather than guessing:

| Indistinguishable | Why |
| --- | --- |
| SDXL 1.0 / Illustrious / Pony / NoobAI / Animagine | identical UNet and text encoder shapes; only the values differ |
| Qwen-Image / Qwen-Image-Edit | same DiT structure |
| Wan 2.x VAE / Qwen-Image VAE | same family of 3D VAE |
| SD1.x VAE / SDXL VAE | same structure |
| FLUX.1 dev / schnell | only the presence of `guidance_in` hints at it |

When metadata survives (`ss_base_model_version`, `modelspec.architecture`), the
tool uses it and says so. When it does not, the filename or the download page is
the only recourse.

## How the rules work

`rules.py` holds the detection rules as data. `stinspect.py` holds the logic. You
should not need to touch the logic to support a new model.

Keys are normalised before matching: the prefix (`lora_unet_`,
`model.diffusion_model.`, `net.`, ...) and the tensor suffix (`.weight`,
`.lora_down.weight`, ...) are stripped and separators unified, so

```
kohya      lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight
diffusers  unet.down_blocks.0.resnets.0.conv1.lora_down.weight
both →     down_blocks_0_resnets_0_conv1
```

One rule then covers every dialect. Adding an architecture is one entry:

```python
{
    "id": "my_model",
    "name": T("My Model", "My Model"),
    "verified": "measured",
    "signals": [
        (r"^some_distinctive_key_\d+", 4, T("why this is evidence", "根拠の説明")),
    ],
    "context_dims": [],     # cross-attention width, if it is fixed
    "veto": [],             # patterns that rule this candidate out
    "note": T("", ""),      # limits worth stating
    "comfy_dir": "diffusion_models",
},
```

To see the key names of a file you want to write a rule for:

```bash
python tools/probe_header.py path/to/file.safetensors
```

## Verification

Rules are checked against real files, not from memory. `tools/verify_rules.py`
fetches only the headers of 27 public models over HTTP Range — no weights
are downloaded and nothing is redistributed — and asserts the expected
classification for each:

```bash
python tools/verify_rules.py
```

The models used, the fingerprints they established, and what remains unverified
are all written up in [docs/key-reference.md](docs/key-reference.md).

Five of those live in gated repositories (FLUX.1, SD3.5). Running the tool
without an account skips them and the other 22 still pass. To include them,
accept the licences on Hugging Face yourself and set `HF_TOKEN`.

## Reading the output

[docs/guide.md](docs/guide.md) covers what each line means, how rank and alpha
combine into the applied scale, what to do when a file is unidentified, and how to
work out why a LoRA is not taking effect.

## License

MIT. See [LICENSE](LICENSE).
