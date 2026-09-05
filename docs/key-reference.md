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
| `Comfy-Org/HunyuanVideo_repackaged` | `split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors` | Tencent Hunyuan Community |
| `Comfy-Org/HunyuanVideo_repackaged` | `split_files/vae/hunyuan_video_vae_bf16.safetensors` | Tencent Hunyuan Community |
| `Wan-AI/Wan2.2-TI2V-5B` | `diffusion_pytorch_model-00001-of-00003.safetensors` | Apache-2.0 |
| `lodestones/Chroma` | `chroma-unlocked-v16.safetensors` | Apache-2.0 |
| `HiDream-ai/HiDream-I1-Full` | `transformer/diffusion_pytorch_model-00001-of-00007.safetensors` | MIT |
| `genmo/mochi-1-preview` | `dit.safetensors` | Apache-2.0 |
| `Tongyi-MAI/Z-Image-Turbo` | `transformer/diffusion_pytorch_model-00001-of-00003.safetensors` | Apache-2.0 |
| `Lightricks/LTX-Video` | `ltxv-13b-0.9.8-dev.safetensors` | LTXV Open Weights |
| `Lightricks/LTX-Video` | `ltxv-13b-0.9.7-distilled-lora128.safetensors` | LTXV Open Weights |
| `THUDM/CogVideoX-5b` | `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | CogVideoX License |
| `THUDM/CogVideoX-5b` | `vae/diffusion_pytorch_model.safetensors` | CogVideoX License |
| `Efficient-Large-Model/Sana_1600M_1024px_diffusers` | `transformer/diffusion_pytorch_model.safetensors` | Apache-2.0 |
| `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | `transformer/diffusion_pytorch_model.safetensors` | OpenRAIL++ |
| `fal/AuraFlow-v0.3` | `aura_flow_0.3.safetensors` | Apache-2.0 |
| `Alpha-VLLM/Lumina-Image-2.0` | `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | Apache-2.0 |
| `Kwai-Kolors/Kolors` | `unet/diffusion_pytorch_model.fp16.safetensors` | Apache-2.0 |
| `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers` | `transformer/diffusion_pytorch_model.safetensors` | Tencent Hunyuan Community |
| `stabilityai/stable-cascade` | `comfyui_checkpoints/stable_cascade_stage_c.safetensors` | Stability AI NC Research Community |
| `stabilityai/stable-cascade` | `comfyui_checkpoints/stable_cascade_stage_b.safetensors` | Stability AI NC Research Community |
| `mit-han-lab/svdq-int4-flux.1-dev` | `transformer_blocks.safetensors` | FLUX.1 [dev] Non-Commercial (inherited) |

The Stable Cascade licence is non-commercial and permits "research or
non-commercial purposes"; it places no restriction on inspection. The Nunchaku
build declares the upstream FLUX.1 [dev] licence and links to it.

The LTX-Video and CogVideoX entries declare "other" rather than a named licence,
so both were read before use. Neither restricts inspection or publishing what you
find. CogVideoX does carry use-based restrictions and requires registration for
commercial use — relevant to running the model, not to reading its header.

The HunyuanVideo files are a ComfyUI repackage that declares the upstream Tencent
licence and links to it. That licence carries a territorial restriction (it does
not extend to the EU, the UK or South Korea) — worth knowing if you intend to use
the model itself. It places no restriction on inspecting it, and in fact
encourages publishing what you find.

Repositories with no declared licence were skipped, even where they held exactly
the layout that was wanted.

These five are gated. They were checked with an account that had accepted their
licences; without one, `tools/verify_rules.py` reports them as unreachable and the
other 42 still run.

| Repository | File | License |
| --- | --- | --- |
| `black-forest-labs/FLUX.1-schnell` | `flux1-schnell.safetensors` | Apache-2.0 |
| `black-forest-labs/FLUX.1-schnell` | `transformer/diffusion_pytorch_model-00001-of-00003.safetensors` | Apache-2.0 |
| `black-forest-labs/FLUX.1-schnell` | `ae.safetensors` | Apache-2.0 |
| `black-forest-labs/FLUX.1-dev` | `flux1-dev.safetensors` | FLUX.1 [dev] Non-Commercial |
| `stabilityai/stable-diffusion-3.5-medium` | `sd3.5_medium.safetensors` | Stability AI Community |

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
| `lora_unet_` / `lora_te*_` | `.lokr_w1`, `.lokr_w2`, `.alpha` | LyCORIS LoKr (measured) |
| `lllite_unet_` | `.conditioning1.N`, `.down.N`, `.mid.N`, `.up.N` | kohya ControlNet-LLLite (measured) |
| any | `.hada_w*`, `.oft_blocks`, `.on_input`, `.diff` | other LyCORIS variants (derived from the LyCORIS source) |

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
the two never compete. Which tensor carries the rank varies by dialect —
`quant_conv`, `conv1`, or `decoder.conv_in.conv` — so the condition names all
three and applies to whichever exists.

**The latent width sits on a different axis depending on the tensor.**
`post_quant_conv` is `[latent, latent, ...]`, so the width is axis 0.
`decoder.conv_in.conv` is `[out, latent, ...]`, so it is axis 1. Folding both
into one pattern made the check depend on which key happened to be found first,
which silently broke the HunyuanVideo VAE when CogVideoX support was added. They
are separate conditions now.

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

### FLUX.1 vs Qwen-Image — the same attention, different surroundings

In the diffusers layout these two are dangerously close. Their attention
submodules are identical, down to the names, and both are 3072 wide:

```
attn.add_k_proj  attn.add_q_proj  attn.add_v_proj  attn.to_add_out
attn.norm_added_k  attn.norm_added_q  attn.norm_k  attn.norm_q
attn.to_k  attn.to_q  attn.to_v  attn.to_out.0
```

An earlier version of these rules classified FLUX's diffusers release as
Qwen-Image on exactly this overlap. What separates them is everything around the
attention:

| | FLUX.1 | Qwen-Image |
| --- | --- | --- |
| feed-forward | `ff`, `ff_context` | `img_mlp`, `txt_mlp` |
| modulation | `norm1.linear`, `norm1_context.linear` | `img_mod.1`, `txt_mod.1` |
| top level | `context_embedder` `[3072, 4096]`, `x_embedder` | `img_in`, `txt_in`, `txt_norm` |
| second stack | `single_transformer_blocks` | none |

`context_embedder` is `[3072, 4096]` because it takes T5-XXL's 4096-wide output —
a width Qwen-Image never has. The rules veto each other on these keys.

FLUX's single-file release is a different layout again (`double_blocks`,
`single_blocks`, `img_in`, `txt_in`), and note that `img_in` / `txt_in` collide
with Qwen-Image's top-level names. They are only weak evidence for that reason.

### HunyuanVideo — Flux's layout with one addition

HunyuanVideo is architecturally close to FLUX's single-file release. These are
shared, name for name:

```
double_blocks.N  single_blocks.N  img_in  txt_in  time_in  vector_in
guidance_in  final_layer
```

Two things tell them apart:

| | HunyuanVideo | FLUX |
| --- | --- | --- |
| text input | `txt_in.individual_token_refiner.blocks.N`, `txt_in.c_embedder`, `txt_in.t_embedder`, `txt_in.input_embedder` | `txt_in.weight` — a bare linear |
| image input | `img_in.proj.weight` | `img_in.weight` |

The token refiner is the decisive one, and the FLUX rule vetoes on it.

The ComfyUI repackage nests everything one level deeper than expected:

```
model.model.double_blocks.0.img_attn.qkv.weight
```

That `model.model.` prefix is why this rule silently matched nothing before it
was ever tested against a file. The prefix is now stripped along with the others.

### FLUX autoencoder

`ae.safetensors` is an ordinary LDM-named 2D VAE (`encoder.down.N.block.M`), so it
matches the same rule as the SD-family VAEs. The latent width differs (16 rather
than 4) but the structure does not.

### SD3.5 single-file

`sd3.5_medium.safetensors` contains the MMDiT (`joint_blocks`) **and** the VAE,
but no text encoders — those are published separately because they are large and
shared. That combination gets its own file kind, `backbone_vae`, so the tool does
not tell you a VAE is missing when it is right there.

### Newer DiTs

Each of these is distinctive enough that one key settles it.

| Architecture | Decisive key | What it is |
| --- | --- | --- |
| Chroma | `distilled_guidance_layer.*` | retrained from FLUX.1 schnell; keeps `double_blocks` / `single_blocks` / `img_in` / `txt_in` but replaces the guidance embedding entirely. The FLUX rule vetoes on this key |
| HiDream-I1 | `double_stream_blocks.N.block.ff_i.experts.N` | mixture-of-experts feed-forward; every attention projection is duplicated with a `_t` suffix for the text stream |
| Mochi 1 | `blocks.N.attn.qkv_x` / `qkv_y`, `t5_y_embedder` | two parallel streams named `_x` (image) and `_y` (text) throughout |
| Z-Image | `noise_refiner.N`, `context_refiner.N`, `cap_embedder` | separate refiner stacks feeding a shared `layers.N` stack |
| LTX-Video | `patchify_proj`, `scale_shift_table` per block | one shared `adaln_single` stage instead of per-block linear modulation. Cross-attention is 4096 wide (T5-XXL) |
| CogVideoX | `patch_embed.text_proj` | the text stream is folded into the patch embedder; a single `attn1` stack, no `attn2` |
| AuraFlow | `double_layers.N.attn.w1q` … `w2o`, `register_tokens` | projections named `w1q`/`w2k` rather than `to_q`/`to_k`. The single-file release bundles the VAE and the T5 encoder too |
| SANA | `caption_norm` | otherwise the PixArt stage; cross-attention 2240 |
| PixArt-alpha / Sigma | `adaln_single` + `pos_embed.proj`, no `caption_norm` | cross-attention 1152 |
| Lumina-Image 2.0 | `time_caption_embed.caption_embedder` | otherwise the same refiner layout as Z-Image |

### Quantisation formats

| Format | Marker | Notes |
| --- | --- | --- |
| fp8 scaled (ComfyUI) | a top-level `scaled_fp8` key, `.scale_weight` / `.scale_input` beside each weight | the weight itself is `F8_E4M3` |
| fp8 plain | `F8_*` dtypes with no scale tensors | |
| SVDQuant (Nunchaku) | `.qweight` + `.wscales`, plus `.smooth` / `.smooth_orig` / `.wzeros` | INT4 packed into `I32`, with a low-rank correction |

**SVDQuant carries `lora_down` and `lora_up` keys, and they are not a LoRA.**
They are the low-rank half of the quantisation scheme. They happen not to collide
with the LoRA dialects because those require a `.weight` suffix
(`.lora_down.weight`) which SVDQuant does not have — a narrow escape rather than
a designed one. Worth remembering if the LoRA patterns are ever loosened.

### Two unrelated things called Hunyuan

`HunyuanDiT` and `HunyuanVideo` share a name and a licence but not an
architecture. HunyuanDiT is `blocks.N` with `attn1`/`attn2` and a
`text_embedding_padding`; HunyuanVideo is Flux-shaped with `double_blocks` and a
token refiner.

### Stable Cascade

Stages B and C are separate files that both classify as full checkpoints — each
bundles a text encoder and a VAE stage. Shared markers are `clip_mapper` /
`clip_img_mapper`, `clf`, and `channelwise` / `depthwise` blocks rather than a
UNet resnet stack. `clip_img_mapper` means Stage C, `effnet_mapper` means Stage B,
and `modelspec.architecture` states it outright. Both are needed to generate:
Stage C makes the small latent, Stage B decodes it.

### Three families that overlap on `adaln_single`

PixArt, SANA and LTX-Video all carry `adaln_single` and `caption_projection`.
They differ in how the image is patchified and whether the caption is normalised:

| | patchify | `caption_norm` |
| --- | --- | --- |
| PixArt | `pos_embed.proj` | no |
| SANA | `pos_embed.proj` | yes |
| LTX-Video | `patchify_proj` | no |

Each rule vetoes on the others' markers.

### Z-Image and Lumina-Image 2.0

Nearly the same file. Both have `layers.N`, `noise_refiner.N` and
`context_refiner.N` with matching submodule names. The caption stage is the
difference:

| | caption stage | embedder / head |
| --- | --- | --- |
| Z-Image | `cap_embedder`, `cap_pad_token` | `all_x_embedder`, `all_final_layer` |
| Lumina 2.0 | `time_caption_embed.caption_embedder` | `x_embedder` |

The Z-Image rule was written first and matched Lumina on the refiner stacks
alone. They veto each other now.

### Kolors is not separable from SDXL

Kolors reuses the SDXL UNet wholesale and projects its ChatGLM text encoder down
to the same 2048-wide cross-attention, so its UNet matches the SDXL rule on every
count — `add_embedding`, block layout, cross-attention width. There is no key
that distinguishes them.

This matters in practice: the tool will say SDXL, and a Kolors UNet will not work
with a CLIP text encoder. The SDXL rule's caveat says so. This is recorded as an
expected result in the verification run rather than papered over.

LTX-Video's single-file releases bundle the VAE under a bare `vae.` prefix, so
they classify as backbone + VAE. That prefix is now stripped along with the
others.

**Wan 2.2 needs no new rule.** The TI2V-5B release has the same structure as
Wan 2.1 — `blocks.N.cross_attn.norm_q/norm_k`, single-letter q/k/v/o
projections, `patch_embedding`, `time_projection` — and the existing rule
identifies it unchanged.

Z-Image is the reason `layers` was added to the backbone-block component rule.
That is safe because a text encoder's layers normalise to `model_layers_N`: the
bare `model.` prefix is not stripped, so `^layers_\d+` cannot reach them.

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

## Three keys that never existed

Reading LyCORIS turned up three keys this project had been matching against that
appear nowhere in the implementation:

| Invented key | Where it was used | What is actually there |
| --- | --- | --- |
| `oft_diag` | OFT rule | `oft_blocks`, `rescale`, `alpha` |
| `boft_*` | OFT rule, for the butterfly variant | nothing — `boft_b` and `boft_m` are Python attributes, never serialised |
| `ia3_weight` | (IA)^3 rule | a parameter named `weight`, plus an `on_input` buffer |

All three came from writing rules that sounded plausible instead of checking. They
would never have matched anything, and the failure mode is silent: the file comes
back unidentified rather than raising an error.

This is the reason for the `unverified` tag and its runtime marker. A rule nobody
has checked is a guess, and it should look like one.

## A trap worth naming

Normalised keys have `.weight` and `.bias` stripped, so `quant_conv.weight`
becomes `quant_conv`. Several rules were originally written as `^quant_conv_`
with a trailing underscore and silently never matched. The symptom is a file
being reported as unidentified rather than an error, which makes it easy to miss.

Patterns must end with `$` or a real separator — never a bare trailing `_`.

## Nothing left unverified

Every rule is now either `measured` against a real file or `derived` from a
primary source. The tally today is 47 measured, 6 derived, 0 unverified, checked against 47 published files.

That will not stay true — new architectures arrive faster than they can be
checked. When you add a rule without a file to test it against, tag it
`unverified` and it will say so at runtime. The three invented keys above are
what happens when that discipline slips.

## Read from the implementation, not from a file (`derived`)

No LyCORIS sample with a declared licence turned up on the Hub, so the key names
were read out of LyCORIS itself rather than guessed. Module paths below are
relative to `lycoris/modules/`.

**LoHa** (`loha.py`) — `custom_state_dict()` emits `alpha`, `dora_scale`,
`hada_w1_a`, `hada_w1_b`, `hada_w2_a`, `hada_w2_b`, `hada_t1`, `hada_t2`.
`hada_w1_b` is `(rank, in_dim)` in both plain and Tucker mode, so the rank is read
from its first axis; `hada_w1_a` flips between `(out, rank)` and `(rank, out)`
depending on mode and is not safe to read a rank from. `hada_t1` / `hada_t2`
present means Tucker decomposition.

**GLoRA** (`glora.py`) — `a1`, `a2`, `b1`, `b2` are `nn.Module`s, so their weights
serialise as `a1.weight` and so on; `bm.weight` appears under Tucker. `a1.weight`
is `(rank, in_dim)`, which is where the rank comes from. Plus an `alpha` buffer.

**OFT and BOFT** (`diag_oft.py`, `boft.py`) — both register `oft_blocks`, an
optional `rescale`, and an `alpha` buffer, and nothing else. They differ only in
tensor rank:

| Variant | `oft_blocks` shape |
| --- | --- |
| DiagOFT | `(block_num, block_size, block_size)` — 3 dims |
| BOFT | `(m, block_num, block_size, block_size)` — 4 dims |

**(IA)^3** (`ia3.py`) — exactly two entries: a parameter literally named `weight`
and a buffer named `on_input`. Since a bare `.weight` matches nearly everything,
`on_input` is the only usable marker.

**full / diff** (`full.py`) — `custom_state_dict()` writes `diff`, and `diff_b`
when the target module had a bias. It stores the delta from the original weight,
not the weight, which is why these files are large.

**DyLoRA** (`dylora.py`) — serialises to `alpha`, `lora_up.weight` and
`lora_down.weight`: **the same key names as a plain LoRA.** It keeps blocks
separately in memory and concatenates them on save. There is therefore no way to
tell a DyLoRA from an ordinary LoRA by its keys, and this project does not try.
The same goes for LyCORIS LoCon.

**SD2.x cross-attention width 1024** — 768 and 2048 are both measured and the
mechanism (the text encoder's output width) is identical, so 1024 for OpenCLIP-H
follows without needing a file. The same reasoning covers Textual Inversion for
SD2.x.

If you have any of these as safetensors, run `tools/probe_header.py` on the file
and open an issue or a PR with the key structure. Promoting a rule from
`unverified` to `measured` is the most useful contribution this project can
receive.
