# Reading the output

What each line means, and what to do with it.

日本語版: [guide.ja.md](guide.ja.md)

## 1. Three lines are usually enough

```
  Type        <- what this file is
  Base        <- what it has to be paired with
  Place in    <- where it goes
```

Those three get the file usable. For a LoRA, add **Strength**.

The rest (Contents, Dialect, Quant, Metadata) is for working out why something
is not loading.

## 2. Line by line

### Type

| Value | Meaning | Works alone? |
| --- | --- | --- |
| Full checkpoint | UNet/DiT + text encoder + VAE, all present | **yes** |
| Diffusion backbone + VAE | the model and its VAE, no text encoder | no — text encoders needed |
| Diffusion backbone only | just the UNet or DiT | no — text encoder and VAE needed |
| LoRA (...layout) | a delta, applied on top of a base | no — needs its base |
| ControlNet | a control model | no — runs alongside a base |
| VAE only | just the latent-to-image decoder | no — attach to a model |
| Text encoder only | just the prompt encoder | no — attach to a model |
| Textual Inversion / embedding | a single learned word | no — called from the prompt |

The distinction between the first three matters when shopping for missing parts.
SD3.5 and some FLUX releases bundle the VAE but leave the text encoders out,
because those are large and shared between models — that is "backbone + VAE".

For a LoRA the layout name follows, and that matters — see **Dialect** below.

### Base and confidence

| Confidence | Meaning | How to treat it |
| --- | --- | --- |
| high | a structural match such as the cross-attention width | trust it |
| medium | several distinctive key names matched | almost certainly right; revisit if it fails to load |
| low | a weak match | a hint, not a finding |
| not identified | no rule matched | investigate (below) |

A marker may follow the name:

| Marker | Meaning |
| --- | --- |
| *(none)* | the rule was checked against a real file |
| `[derived, not directly measured]` | taken from a primary source — it follows from a measured fact, or the key names were read out of the implementation that writes them |
| `[unverified / inferred]` | a guess, from key naming seen second-hand. **Can be wrong.** |

Today no rule is `unverified`, but that changes as new architectures appear.

`Evidence:` lines say why. `Caveat:` lines say where the answer stops:

```
  Base        SDXL family (SDXL 1.0 / Illustrious / Pony / NoobAI / Animagine ...)   confidence high
              Evidence: cross-attention input width = 2048
              Caveat: Derivatives such as Illustrious, Pony, NoobAI and Animagine
                      are structurally identical to SDXL 1.0 ...
```

Read that as: **it is definitely SDXL-family; which member is unknowable from the
file.** In practice, try your SDXL checkpoints in turn. Guessing "anime style, so
probably Illustrious" is inference from the filename or the source page, not
something the tool established.

### Contents

What is inside. `UNet (LDM / SAI naming)` versus `UNet (diffusers naming)` is a
difference in key naming conventions, not in capability.

### Targets

Where a LoRA applies.

- `UNet / DiT` — style, composition, subject. Most LoRAs.
- `Text encoder` — changes how particular words are interpreted; common for LoRAs
  with trigger words.
- Both — the full effect needs both connected.

If a LoRA targets the text encoder, connect the CLIP side too. ComfyUI's
`LoraLoaderModelOnly` skips it, and the LoRA will then underperform.

### Strength (rank and alpha)

```
  Strength    rank 16   alpha 16
```

- **rank** (`network_dim`) — capacity. Higher means more expressive and larger.
- **alpha** — a scaling constant fixed at training time.

The factor actually applied is `alpha / rank`. Both the training side and the
inference side use the same formula.

Training, kohya `networks/lora.py`:

```python
alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
self.scale = alpha / self.lora_dim
```

Inference, ComfyUI `comfy/weight_adapter/lora.py`:

```python
if v[2] is not None:
    alpha = v[2] / mat2.shape[0]
else:
    alpha = 1.0
...
weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
```

So the final factor is **UI strength × (alpha / rank)**.

| Shown | alpha / rank | Reading |
| --- | --- | --- |
| rank 16 / alpha 16 | 1.0 | standard; UI strength applies as-is |
| rank 64 / alpha 8 | 0.125 | deliberately restrained |
| rank 8 / alpha 1 | 0.125 | restrained |
| *(no alpha shown)* | 1.0 | PEFT-style files carry no alpha; loaders use 1.0 |

This describes how it was trained, not a recommended strength. Follow the
distributor's guidance if there is any.

`mixed rank: 8 (743 layers), 4 (346 layers)` means the rank varies per block.
That is a legitimate training choice, not damage.

**No rank shown for LoKr.** LyCORIS LoKr factorises differently per module, so
there is no single rank to report — `ss_network_dim` in its metadata is usually a
placeholder rather than a real value. Alpha is still shown.

### Dialect

`LDM naming and diffusers naming are both present` — one file carrying two key
sets. Usually loads fine.

`LoRA (dot-separated diffusers layout / OneTrainer internal save)` deserves
attention: the keys are not in the `lora_unet_` form, so A1111 and ComfyUI may
refuse it. With OneTrainer, use the file written by **save**, not one from a
`backup` folder.

`ControlNet-LLLite` is not a LoRA and not an ordinary ControlNet. It injects
conditioning into the UNet's attention and needs a loader that supports it
specifically — ComfyUI has a dedicated node; ordinary ControlNet loaders will
not take it.

Note that LyCORIS LoCon and DyLoRA serialise to exactly the same key names as a
plain LoRA, so a file reported as `LoRA (kohya / sd-scripts layout)` may be any
of the three. Nothing in the header distinguishes them.

### Quant

`fp8 scaled (ComfyUI layout)` — weights in fp8 with per-layer correction factors.
Less VRAM, but the loader has to support it.

`SVDQuant / Nunchaku` — INT4 weights with a low-rank correction. Needs the
Nunchaku nodes specifically. Note that these files contain `lora_down` and
`lora_up` keys that are **not** a LoRA — they are part of the quantisation.

### Metadata

Written at training time, and often **more reliable than the structural
verdict**. Two entries settle the base model outright:

- `training base` (`ss_base_model_version`) — the actual base used for training
- `declared architecture` (`modelspec.architecture`) — what the producing tool declared

`merged from` tells you what went into a merge.

### Triggers

Most frequent training tags — candidate words to put in the prompt. Frequency is
not the same as a trigger; generic tags like `1girl` also rank high.

### Place in

The ComfyUI folder. ComfyUI maps the legacy names internally
(`unet` → `diffusion_models`, `clip` → `text_encoders`), so either works.

## 3. What to do per type

### Full checkpoint

Drop it in `models/checkpoints` and load with Load Checkpoint. Done.
A1111 and Forge use `models/Stable-diffusion`.

### Diffusion backbone only

`models/diffusion_models`. **Will not run on its own.** Check the Base line and
gather the matching parts:

| Base | Also needed |
| --- | --- |
| Qwen-Image / Qwen-Image-Edit | Qwen2.5-VL 7B text encoder + Qwen-Image VAE |
| FLUX.1 | T5-XXL + CLIP-L + the Flux VAE (`ae.safetensors`) |
| SD3 / SD3.5 | CLIP-L + CLIP-G + T5-XXL, and the VAE if not bundled |
| Wan 2.x | the matching text encoder + 3D VAE |
| HunyuanVideo | the LLaVA-Llama3 text encoder + CLIP-L + the HunyuanVideo 3D VAE |
| Anima | a Qwen3-0.6B class text encoder + the matching VAE |
| Chroma | the same parts as FLUX.1 — T5-XXL + CLIP-L + the Flux VAE |
| HiDream-I1 | CLIP-L + CLIP-G + T5-XXL + a Llama text encoder |
| Mochi 1 | T5-XXL + the Mochi VAE |
| Z-Image | its own text encoder + VAE, published alongside it |
| LTX-Video | T5-XXL; the single-file releases already contain the VAE |
| CogVideoX | T5-XXL + the CogVideoX 3D VAE |
| PixArt / SANA | T5 for PixArt, a Gemma encoder for SANA |
| AuraFlow | T5; the single-file release already contains everything |
| Lumina-Image 2.0 | its own Gemma-class encoder + VAE |
| HunyuanDiT | a bilingual CLIP + mT5 (unrelated to HunyuanVideo) |

To find out what you already have, scan the whole folder:

```bash
python stinspect.py "path/to/ComfyUI/models" -r -o inventory.txt
```

Entries typed `Text encoder only` and `VAE only` whose Base line agrees are the
ones to pair it with.

### Diffusion backbone + VAE

`models/checkpoints`, loaded with Load Checkpoint — but the text encoders still
have to be supplied separately (TripleCLIPLoader for SD3.5, DualCLIPLoader for
FLUX). The VAE is already inside, so there is no need to hunt for one.

### LoRA

`models/loras`, loaded with LoraLoader. A mismatched base means either no effect
or a broken image. If loading errors out, check the Dialect line.

### VAE / text encoder

`models/vae` and `models/text_encoders`, attached to a model. Swapping the VAE
changes colour and fine detail. If the checkpoint already bundles one, the
built-in VAE is used unless you override it.

## 3b. The summary at the end of a folder scan

Scanning more than one file ends with a summary:

```
==============================================================================
Summary - 40 file(s)

By type
    12  LoRA (kohya / sd-scripts layout)
    10  Diffusion backbone only ...
     3  Full checkpoint ...

By base model
    17  SDXL family ...
    10  not identified
     3  Qwen-Image / Qwen-Image-Edit

Not identified - 10 file(s), listed for next time
  mystery.safetensors
      top-level keys: blocks (690), t_embedder (4), input_layer (2)
```

The last block is the one to act on. Those files need a rule, and the top-level
key names are the first thing you would need to write one.

To keep that list rather than scroll back to it:

```bash
python stinspect.py path/to/models -r --unresolved todo.txt
```

That file carries paths, tensor counts, dtypes, sample keys and any weak matches
— everything needed to add a rule later, or to hand to someone who will.

Models outside image and video generation (depth estimators, 3D generators,
vision backbones) show up in this list too. They are out of scope, not a gap.

`--no-summary` turns the block off.

Nothing is moved, renamed or written over at any point. The tool reports.

## 4. When nothing is identified

```
  Base        not identified
              weak match (not enough to conclude): Wan 2.x (2: ...)
```

**Step 1 — read all the metadata.**

```bash
python stinspect.py "mystery.safetensors" --meta -o meta.txt
```

`ss_sd_model_name`, `ss_network_args`, or a ComfyUI `workflow` blob often survive
and name the base outright.

**Step 2 — look at the key names.**

```bash
python stinspect.py "mystery.safetensors" --keys
```

The leading segment (`double_blocks`, `joint_blocks`, `blocks`, ...) is the
architecture's fingerprint. Searching for that string usually names the model.

For the full structure:

```bash
python tools/probe_header.py "mystery.safetensors"
```

**Step 3 — add a rule.** One entry in `rules.py` and it is recognised from then
on. A pull request adding it is welcome — see [key-reference.md](key-reference.md).

Models outside image generation (DINOv3, MoGe, 3D generators) come out
unidentified by design; the scope here is image and video generation.

## 5. Traps

### Size mismatch reported

```
  [!] Size mismatch: expected 6937890000, actual 4200000000
      - the download may be incomplete or the file corrupt
```

The header parsed but the tensor data is short. Download it again.

### It says SDXL but the images come out wrong

Usually the derivative. Illustrious, Pony and NoobAI share SDXL's structure but
not its training, so a Pony LoRA on an Illustrious checkpoint underperforms or
breaks. The tool cannot separate them.

Check `training base` or `merged from` in the metadata. Failing that, the source page.

### A LoRA has no effect

In order:

1. Wrong base — compare the Base line against the checkpoint in use
2. Unsupported dialect — check for `dot-separated diffusers layout`
3. It targets the text encoder but CLIP is not connected — check Targets
4. Missing trigger word — check Triggers

### Many near-identical files

Trainers emit checkpoints every few epochs. A CSV makes the duplicates obvious:

```bash
python stinspect.py "path/to/models" -r --csv all.csv
```

Matching `tensors` and `params` usually means different steps of one run.
