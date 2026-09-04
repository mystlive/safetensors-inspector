# Key reference

What every detection rule in `rules.py` is based on.

Rules carry one of three confidence tags:

| Tag | Meaning | Printed as |
| --- | --- | --- |
| `measured` | checked against a real file's header | *(nothing)* |
| `derived` | taken from a primary source — either it follows from a measured fact, or the key names were read out of the implementation that writes them | `[derived, not directly measured]` |
| `unverified` | inferred from key naming seen second-hand | `[unverified / inferred]` |

## How this was measured

Headers were fetched with HTTP Range requests against Hugging Face — the first
8 bytes to learn the header length, then exactly that many bytes of JSON. No
weights were downloaded, and nothing was redistributed. A 5 GB checkpoint costs
about 80 KB to inspect this way.

`tools/verify_rules.py` reproduces the whole run:

```bash
python tools/verify_rules.py
```

It reports `OK / NG` per model against the expected classification, so a rule
change that breaks an earlier finding shows up immediately.

## Format specification

- Binary layout: the Format section of <https://github.com/huggingface/safetensors>
  - "8 bytes: `N`, an unsigned little-endian 64-bit integer, containing the size of the header"
  - "A special key `__metadata__` is allowed to contain free form string-to-string map"
  - Each tensor entry carries `dtype`, `shape` and `data_offsets`
- Dtype names and bit widths: `safetensors/src/tensor.rs`, the `Dtype` enum and `bitsize()`
  - `F4` 4; `F6_E2M3`/`F6_E3M2` 6; `BOOL`/`U8`/`I8`/`F8_*` 8;
    `I16`/`U16`/`F16`/`BF16` 16; `I32`/`U32`/`F32` 32; `F64`/`I64`/`U64`/`C64` 64

## Models checked

All of these are ungated, so the run needs no account and no token.

| Repository | File | License |
| --- | --- | --- |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | `v1-5-pruned-emaonly.safetensors` | CreativeML Open RAIL-M |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | `unet/diffusion_pytorch_model.fp16.safetensors` | CreativeML Open RAIL-M |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | `text_encoder/model.fp16.safetensors` | CreativeML Open RAIL-M |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | `vae/diffusion_pytorch_model.fp16.safetensors` | CreativeML Open RAIL-M |
| `stabilityai/stable-diffusion-xl-base-1.0` | `sd_xl_base_1.0.safetensors` | OpenRAIL++ |
| `stabilityai/stable-diffusion-xl-base-1.0` | `sd_xl_offset_example-lora_1.0.safetensors` | OpenRAIL++ |
| `stabilityai/stable-diffusion-xl-base-1.0` | `text_encoder_2/model.fp16.safetensors` | OpenRAIL++ |
| `stabilityai/sdxl-vae` | `diffusion_pytorch_model.safetensors` | MIT |
| `latent-consistency/lcm-lora-sdxl` | `pytorch_lora_weights.safetensors` | OpenRAIL++ |
| `diffusers/controlnet-canny-sdxl-1.0` | `diffusion_pytorch_model.fp16.safetensors` | OpenRAIL++ |
| `Qwen/Qwen-Image` | `transformer/diffusion_pytorch_model-00001-of-00009.safetensors` | Apache-2.0 |
| `Qwen/Qwen-Image` | `vae/diffusion_pytorch_model.safetensors` | Apache-2.0 |
| `Wan-AI/Wan2.1-T2V-1.3B` | `diffusion_pytorch_model.safetensors` | Apache-2.0 |
| `comfyanonymous/flux_text_encoders` | `t5xxl_fp16.safetensors` | Apache-2.0 |
| `comfyanonymous/flux_text_encoders` | `t5xxl_fp8_e4m3fn_scaled.safetensors` | Apache-2.0 |
| `comfyanonymous/flux_text_encoders` | `clip_l.safetensors` | Apache-2.0 |
| `LyliaEngine/USNR_STYLE_XL_lokr` | `USNR STYLE_XL_lokr.safetensors` | CDLA-Permissive-2.0 |
| `monster-labs/control_v1p_sd15_qrcode_monster` | `control_v1p_sd15_qrcode_monster.safetensors` | OpenRAIL++ |
| `monster-labs/control_v1p_sd15_qrcode_monster` | `diffusion_pytorch_model.safetensors` | OpenRAIL++ |
| `kohya-ss/controlnet-lllite` | `controllllite_v01032064e_sdxl_canny.safetensors` | Apache-2.0 |

Repositories with no declared licence were skipped, even where they held exactly
the layout that was wanted.

Additional dialects (Anima, Qwen-Image LoRAs, OneTrainer output) were measured
against locally trained and locally held files. Those are described below by
their structure; the files themselves are not identified.

## Fingerprints

### Cross-attention width — the sturdiest signal

The cross-attention `to_k` / `to_v` input width equals the text encoder's output
width. A UNet and a text encoder of different widths cannot be connected at all,
so a match settles the base model on its own.

| Width | Base | Status |
| --- | --- | --- |
| 768 | SD1.x | measured (`attn2.to_v.weight` = `[640, 768]`) |
| 1024 | SD2.x | derived (OpenCLIP-H output width) |
| 2048 | SDXL family | measured (`attn2.to_v.weight` = `[1280, 2048]`) |

### SD1.x checkpoint

- Prefixes `model.diffusion_model.`, `cond_stage_model.transformer.`, `first_stage_model.`
- `token_embedding.weight` = `[49408, 768]`
- No `label_emb`
- Scheduler constants (`alphas_cumprod`, `betas`, ...) ship inside the file

### SDXL checkpoint

- Prefixes `model.diffusion_model.`, `conditioner.embedders.{0,1}.`, `first_stage_model.`
- `conditioner.embedders.0` is CLIP-L, `conditioner.embedders.1` is OpenCLIP-G
- `model.diffusion_model.label_emb.*` exists — SD1.x has no counterpart
- The diffusers-side UNet uses `add_embedding.linear_{1,2}` instead

SDXL derivatives (Illustrious, Pony, NoobAI, Animagine and merges of them) are
byte-for-byte identical in structure to SDXL 1.0. Only the weight values differ.
No amount of header inspection separates them; metadata or the filename is the
only recourse.

### LoRA dialects

Four layouts were observed. Suffixes alone are not enough to tell them apart —
the prefix has to be checked too.

| Prefix | Suffixes | Written by |
| --- | --- | --- |
| `lora_unet_` / `lora_te*_` | `.lora_down.weight`, `.lora_up.weight`, `.alpha` | kohya / sd-scripts, and most distributed LoRAs |
| `unet.` / `te*.` | same as above | OneTrainer's internal backup format |
| `diffusion_model.` | `.lora_A.weight`, `.lora_B.weight` | ai-toolkit, diffusers, PEFT |
| none | `.hada_w1_a`, `.lokr_w1`, ... | LyCORIS variants (unverified) |

The kohya and OneTrainer-internal layouts share the same suffixes. Classifying by
suffix alone mislabels one as the other — this was an actual bug found during
verification.

The same training run saved by OneTrainer twice produced identical tensor counts,
rank and alpha but different key naming: the backup used the diffusers names
(`up_blocks`), the saved output used the LDM names (`input_blocks`). OneTrainer
converts to the kohya layout on save.

### Mixed naming in one file

One LoRA carried both `lora_unet_down_blocks_...` (diffusers naming) and
`lora_unet_output_blocks_...` (LDM naming) in the same file, with rank 8 on some
modules and rank 4 on others, and alpha 4 and 1 respectively. Not a defect;
either a file shipping both key sets for compatibility, or a merge. The tool
reports the mix rather than picking one.

### Qwen-Image DiT

- `transformer_blocks.N` with `txt_mod` / `img_mod` and separate `txt_mlp` / `img_mlp`
- `attn.add_k_proj` / `add_q_proj` / `add_v_proj` / `to_add_out` — the text stream
  gets its own projections
- Hidden width 3072
- Qwen-Image and Qwen-Image-Edit share this structure exactly

### Wan 2.x

- `blocks.N.self_attn.{q,k,v,o}` — projections are single letters, not `q_proj` or `to_q`
- `blocks.N.cross_attn.norm_q` / `norm_k` — note the word order; Anima uses `k_norm`
- `patch_embedding`, `time_embedding`, `time_projection`, `text_embedding`, `head`
- Hidden width 1536 on the 1.3B model

### Anima (DiT with an LLM adapter)

- `blocks.N.adaln_modulation_{self_attn,cross_attn,mlp}` — three modulation paths
- `llm_adapter.blocks.N` — unique to this architecture
- `blocks.0.cross_attn.k_proj.weight` = `[2048, 1024]`, so hidden 2048 with a
  1024-wide text side
- A bare checkpoint uses the `net.` prefix; the same model re-saved by ComfyUI uses
  `model.diffusion_model.` — which is why prefixes are stripped before matching

### VAE: 2D vs 3D

In the diffusers layout both use `encoder.down_blocks` / `decoder.up_blocks`, so
naming does not separate them. Conv rank does:

| | `quant_conv.weight` | Example |
| --- | --- | --- |
| 2D VAE | 4 dims `[out, in, h, w]` | `[8, 8, 1, 1]` |
| 3D VAE | 5 dims `[out, in, t, h, w]` | `[32, 32, 1, 1, 1]` |

The rules express this as `require_ndim`, a hard condition rather than a score, so
the two never compete.

The same Qwen-Image VAE ships under two different namings: the diffusers release
uses `encoder.down_blocks`, the ComfyUI redistribution uses `encoder.downsamples`.
Both are handled.

### Text encoders

| Family | Fingerprint |
| --- | --- |
| CLIP | `text_model.encoder.layers.N` |
| T5 | `encoder.block.N.layer.M.SelfAttention` / `DenseReluDense` |
| Qwen | `model.layers.N.self_attn.{q,k}_norm` plus SwiGLU MLP |
| Qwen2.5-VL | the above plus `visual.blocks.N` and `visual.patch_embed` |

### fp8 quantization (ComfyUI layout)

- A top-level `scaled_fp8` key
- `.scale_weight` and `.scale_input` (F32) beside each quantized weight
- The weight itself in `F8_E4M3`

### LyCORIS LoKr

- Same prefixes as kohya LoRA (`lora_unet_`, `lora_te1_`), different suffixes:
  `.lokr_w1`, `.lokr_w2`, and sometimes `.lokr_w2_a` / `.lokr_w2_b`, plus `.alpha`
- `lokr_w1` is tiny (`[6, 6]`, `[5, 5]`), `lokr_w2` carries the bulk (`[512, 128]`)
- There is no single rank: the factorisation splits differently per module, and
  `ss_network_dim` in the metadata was `100000` — a placeholder, not a rank. The
  tool therefore reports alpha but no rank for LoKr.
- Suffix alone distinguishes it from plain LoRA, since `.lora_down.weight` is absent

### ControlNet

Both layouts measured, on the same model published in both forms:

| Layout | Fingerprint |
| --- | --- |
| LDM (A1111) | `input_blocks.N`, `middle_block`, `zero_convs.N`, `input_hint_block`, `middle_block_out`, no prefix |
| diffusers | `controlnet_cond_embedding.*`, `controlnet_down_blocks.N`, `controlnet_mid_block`, `down_blocks.N` |

Both also match the UNet component rules, since a ControlNet *is* a copy of the
UNet's encoder half. The ControlNet-specific keys are what settle the type.

### Textual Inversion

An embedding is a slab of the text encoder's output width, so the tensor shape
names the base outright.

| File | Tensors | Base |
| --- | --- | --- |
| `boring_sdxl_v1.safetensors` | `clip_g` `[8, 1280]`, `clip_l` `[8, 768]` | SDXL — one vector per encoder |
| `boring_e621_v4.safetensors` | `emb_params` `[8, 768]` | SD1.x |

The first axis is the number of prompt tokens the embedding occupies. SD2.x would
be `emb_params [n, 1024]`; that variant is `derived`, not measured.

Note that TI is mostly distributed as `.pt` / `.bin`. Safetensors embeddings are
the minority, which is why this took a while to find.

### ControlNet-LLLite

A third control format, unrelated to either ControlNet layout:

- Prefix `lllite_unet_`, then the target module path
  (`input_blocks_4_1_transformer_blocks_0_attn1_to_k`)
- Suffixes `.conditioning1.N.weight`, `.down.N.weight`, `.mid.N.weight`, `.up.N.weight`
- `modelspec.architecture` = `stable-diffusion-xl-v1-base/control-net-lllite`
- It injects conditioning into the UNet's attention rather than running a parallel
  encoder, so it needs a loader that supports it specifically — ordinary ControlNet
  loaders will not take it
- The conditioning width is usually encoded in the filename
  (`v01032064e` → 32/64), not recoverable from the header

## A trap worth naming

Normalised keys have `.weight` and `.bias` stripped, so `quant_conv.weight`
becomes `quant_conv`. Several rules were originally written as `^quant_conv_`
with a trailing underscore and silently never matched. The symptom is a file
being reported as unidentified rather than an error, which makes it easy to miss.

Patterns must end with `$` or a real separator — never a bare trailing `_`.

## Not verified

No file at hand, so these rules come from key naming in public implementations
and are marked `unverified`:

| Rule | Why it is still unverified |
| --- | --- |
| FLUX.1 (`double_blocks`, `single_blocks`, `img_in`) | the repository is gated |
| SD3 / SD3.5 (`joint_blocks`) | gated |
| HunyuanVideo (`txt_in_individual_token_refiner`) | no ungated sample found |
| LyCORIS GLoRA (`a1.weight`, `b1.weight`) | not read from the source yet |
| LyCORIS full / diff, (IA)^3 | same |

## Read from the implementation, not from a file (`derived`)

No sample with a declared licence turned up for these, so the key names were
taken from LyCORIS itself rather than guessed.

**LoHa** — `lycoris/modules/loha.py`, `custom_state_dict()` emits `alpha`,
`dora_scale`, `hada_w1_a`, `hada_w1_b`, `hada_w2_a`, `hada_w2_b`, `hada_t1`,
`hada_t2`. `hada_w1_b` is `(rank, in_dim)` in both plain and Tucker mode, so the
rank is read from its first axis; `hada_w1_a` flips between `(out, rank)` and
`(rank, out)` depending on mode and is not safe to read a rank from. The presence
of `hada_t1` / `hada_t2` means Tucker decomposition.

**OFT** — `lycoris/modules/diag_oft.py` registers `oft_blocks`
`(block_num, block_size, block_size)`, an optional `rescale`, and an `alpha`
buffer. An earlier version of this project's rule looked for an `oft_diag` key,
which does not exist anywhere in LyCORIS. BOFT is a separate module and its keys
have not been read.

**SD2.x cross-attention width 1024** — 768 and 2048 are both measured and the
mechanism (the text encoder's output width) is identical, so 1024 for OpenCLIP-H
follows without needing a file.

**Textual Inversion for SD2.x** — same reasoning, applied to `emb_params`.

If you have any of these as safetensors, run `tools/probe_header.py` on the file
and open an issue or a PR with the key structure. Promoting a rule from
`unverified` to `measured` is the most useful contribution this project can
receive.
