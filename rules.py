# -*- coding: utf-8 -*-
"""
Detection rules for stinspect. Data only - the evaluation logic lives in stinspect.py.

To support a new architecture, add one entry here. You should not need to touch
stinspect.py.

--------------------------------------------------------------------------
The "verified" field
--------------------------------------------------------------------------
  "measured"   The rule was checked against a real file's header.
               See docs/key-reference.md for what was checked and where it came from.
  "derived"    Not checked against a real file, but taken from a primary source:
               either it follows from a measured fact (SD1.x cross-attention is
               768 and SDXL is 2048, both measured, so SD2.x at 1024 follows from
               the same mechanism), or the key names were read out of the
               implementation that writes them.
  "unverified" Neither. Inferred from key naming seen second-hand. Findings from
               these rules are printed with an "[unverified / inferred]" marker.

Keep this distinction. Measured facts and guesses must not look alike.

--------------------------------------------------------------------------
Writing patterns
--------------------------------------------------------------------------
Patterns match against the NORMALISED key, which has the prefix
(lora_unet_, model.diffusion_model., ...) and the tensor suffix
(.weight, .bias, .lora_down.weight, ...) stripped, and separators unified to "_".

    kohya      lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight
    diffusers  unet.down_blocks.0.resnets.0.conv1.lora_down.weight
    both ->    down_blocks_0_resnets_0_conv1

Because ".weight" is stripped, a pattern must NOT require a trailing "_".
Write `^quant_conv$`, not `^quant_conv_`.
"""

from i18n import T

# =========================================================================
# 1. Prefixes to strip, and the component each one implies.
#    Measured: SD1.x/SDXL single-file checkpoints use "model.diffusion_model.";
#    a bare DiT uses "net."; ai-toolkit LoRAs use "diffusion_model.";
#    OneTrainer's internal format uses "unet."; kohya uses "lora_unet_".
# =========================================================================
STRIP_PREFIXES = [
    ("lora_unet_", "unet"),
    ("lora_te1_", "text_encoder_1"),
    ("lora_te2_", "text_encoder_2"),
    ("lora_te_", "text_encoder"),
    ("lycoris_unet_", "unet"),
    ("lllite_unet_", "unet"),
    ("lora_transformer_", "unet"),
    ("model.diffusion_model.", "unet"),
    # ComfyUI's repackaged HunyuanVideo nests the model twice: model.model.*
    ("model.model.", "unet"),
    ("diffusion_model.", "unet"),
    ("transformer.", "unet"),
    ("unet.", "unet"),
    ("net.", "unet"),
    ("cond_stage_model.", "text_encoder"),
    ("conditioner.embedders.", "text_encoder"),
    ("first_stage_model.", "vae"),
    # LTX-Video ships the VAE inside the same file under a bare "vae." prefix
    ("vae.", "vae"),
    # AuraFlow's single-file release bundles the text encoder too
    ("text_encoders.", "text_encoder"),
    # Stable Cascade's all-in-one files use the singular form
    ("text_encoder.", "text_encoder"),
    ("te1.", "text_encoder_1"),
    ("te2.", "text_encoder_2"),
    ("te.", "text_encoder"),
]

# =========================================================================
# 2. Adapter dialects - decides whether the file is a delta or a full model.
#    `patterns` match the raw key's suffix; `prefix_pattern`, when present,
#    must also match the start of the key.
# =========================================================================
ADAPTER_DIALECTS = [
    {
        "id": "lora_kohya",
        "name": T("LoRA (kohya / sd-scripts layout)",
                  "LoRA (kohya / sd-scripts 形式)"),
        "prefix_pattern": r"^(lora|lycoris)_(unet|te\d?|transformer)_",
        "patterns": [r"\.lora_down\.weight$", r"\.lora_up\.weight$"],
        "alpha_pattern": r"\.alpha$",
        "down_pattern": r"\.lora_down\.weight$",
        "up_pattern": r"\.lora_up\.weight$",
        "verified": "measured",
        "note": T("The most widely supported LoRA layout. A1111, ComfyUI and Forge read it "
                  "as-is. LyCORIS LoCon and DyLoRA serialise to the same key names, so they "
                  "are indistinguishable from a plain LoRA here.",
                  "A1111 / ComfyUI / Forge がそのまま読める最も一般的な LoRA 形式。"
                  "LyCORIS の LoCon と DyLoRA も同じキー名で保存されるため、"
                  "通常の LoRA とは区別できない"),
    },
    {
        # Measured: OneTrainer writes this under workspace/.../backup/, while its
        # "save" output converts the same weights to the lora_unet_ layout.
        "id": "lora_dotted",
        "name": T("LoRA (dot-separated diffusers layout / OneTrainer internal save)",
                  "LoRA (ドット区切り diffusers 形式 / OneTrainer 内部保存)"),
        "prefix_pattern": r"^(unet|te\d?|text_encoder(_\d)?|vae)\.",
        "patterns": [r"\.lora_down\.weight$", r"\.lora_up\.weight$"],
        "alpha_pattern": r"\.alpha$",
        "down_pattern": r"\.lora_down\.weight$",
        "up_pattern": r"\.lora_up\.weight$",
        "verified": "measured",
        "note": T("Keys are not in the lora_unet_ form, so A1111 and ComfyUI may fail to load it. "
                  "With OneTrainer, use the file written by 'save' rather than one from a backup folder.",
                  "キー名が lora_unet_ 形式でないため、A1111 / ComfyUI がそのままでは読めないことがある。"
                  "OneTrainer なら「保存」側で書き出した方のファイルを使うこと"),
    },
    {
        "id": "lora_peft",
        "name": T("LoRA (PEFT / diffusers layout)",
                  "LoRA (PEFT / diffusers 形式)"),
        "prefix_pattern": None,
        "patterns": [r"\.lora_A\.weight$", r"\.lora_B\.weight$"],
        "alpha_pattern": None,
        "down_pattern": r"\.lora_A\.weight$",
        "up_pattern": r"\.lora_B\.weight$",
        "verified": "measured",
        "note": T("Written by ai-toolkit, diffusers and ComfyUI-native training. "
                  "Usually carries no alpha, so loaders apply a scale of 1.0.",
                  "ai-toolkit / diffusers / ComfyUI ネイティブ LoRA。alpha を持たず"
                  "倍率 1.0 で適用される実装が多い"),
    },
    {
        # Key names taken from LyCORIS itself (lycoris/modules/loha.py,
        # custom_state_dict): alpha, dora_scale, hada_w1_a, hada_w1_b,
        # hada_w2_a, hada_w2_b, hada_t1, hada_t2.
        # hada_w1_b is (rank, in_dim) in both plain and Tucker mode, so its
        # first axis is the rank. hada_w1_a flips between (out, rank) and
        # (rank, out) depending on mode, which is why the rank is read from _b.
        "id": "lycoris_loha",
        "name": T("LyCORIS LoHa (Hadamard product)", "LyCORIS LoHa (アダマール積)"),
        "prefix_pattern": None,
        "patterns": [r"\.hada_w1_a$", r"\.hada_w1_b$",
                     r"\.hada_w2_a$", r"\.hada_w2_b$",
                     r"\.hada_t1$", r"\.hada_t2$"],
        "alpha_pattern": r"\.alpha$",
        "down_pattern": r"\.hada_w1_b$",
        "up_pattern": r"\.hada_w1_a$",
        "verified": "derived",
        "note": T("Needs LyCORIS support (ComfyUI handles it natively). "
                  "hada_t1 / hada_t2 present means Tucker decomposition is in use.",
                  "LyCORIS 拡張が必要（ComfyUI は標準で読める）。"
                  "hada_t1 / hada_t2 があれば Tucker 分解を使っている"),
    },
    {
        # Measured. LoKr has no single "rank": each module carries a small
        # lokr_w1 (e.g. [6, 6]) and a larger lokr_w2, sometimes split into
        # lokr_w2_a / lokr_w2_b. Reporting a rank would be misleading, so
        # down_pattern is left unset.
        "id": "lycoris_lokr",
        "name": T("LyCORIS LoKr (Kronecker product)", "LyCORIS LoKr (クロネッカー積)"),
        "prefix_pattern": None,
        "patterns": [r"\.lokr_w1$", r"\.lokr_w1_a$", r"\.lokr_w1_b$",
                     r"\.lokr_w2$", r"\.lokr_w2_a$", r"\.lokr_w2_b$",
                     r"\.lokr_t2$"],
        "alpha_pattern": r"\.alpha$",
        "down_pattern": None,
        "up_pattern": None,
        "verified": "measured",
        "note": T("Needs LyCORIS support (ComfyUI handles it natively). "
                  "The factorisation has no single rank, so none is reported; "
                  "ss_network_dim in the metadata is often a placeholder.",
                  "LyCORIS 拡張が必要（ComfyUI は標準で読める）。"
                  "分解の都合で単一の rank が存在しないため表示しない。"
                  "メタデータの ss_network_dim は便宜的な値であることが多い"),
    },
    {
        # Measured against kohya-ss/controlnet-lllite. Not a LoRA and not a
        # ControlNet backbone: it injects conditioning into the UNet's attention,
        # so it is classified as a control model.
        "id": "controlnet_lllite",
        "name": T("ControlNet-LLLite (SDXL)", "ControlNet-LLLite (SDXL)"),
        "kind": "controlnet",
        "prefix_pattern": r"^lllite_unet_",
        "patterns": [r"\.conditioning1\.\d+\.weight$", r"\.(down|mid|up)\.\d+\.weight$"],
        "alpha_pattern": None,
        "down_pattern": None,
        "up_pattern": None,
        "verified": "measured",
        "note": T("kohya's ControlNet-LLLite. Needs a loader that supports it "
                  "(ComfyUI has a dedicated node); ordinary ControlNet loaders will "
                  "not take it. The conditioning width is usually encoded in the "
                  "filename, e.g. v01032064e means 32/64.",
                  "kohya の ControlNet-LLLite。対応ローダーが必要"
                  "（ComfyUI は専用ノードがある）。通常の ControlNet ローダーでは読めない。"
                  "条件付けの次元はファイル名に入っていることが多い（v01032064e なら 32/64）"),
    },
    {
        # lycoris/modules/full.py, custom_state_dict(): stores "diff" and,
        # when the target had one, "diff_b" - the delta from the original
        # weight rather than the weight itself.
        "id": "lycoris_full",
        "name": T("LyCORIS full / diff (full delta)", "LyCORIS full / diff (全差分)"),
        "prefix_pattern": None,
        "patterns": [r"\.diff$", r"\.diff_b$"],
        "alpha_pattern": None,
        "down_pattern": None,
        "up_pattern": None,
        "verified": "derived",
        "note": T("Stores the delta itself rather than a low-rank approximation, so the file is large.",
                  "低ランク近似ではなく差分そのものを持つ。ファイルサイズが大きい"),
    },
    {
        # lycoris/modules/glora.py: a1/a2/b1/b2 are nn.Modules, so their weights
        # land as a1.weight etc. bm.weight appears under Tucker decomposition.
        # a1.weight is (rank, in_dim), so the rank is its first axis.
        "id": "lycoris_glora",
        "name": T("LyCORIS GLoRA", "LyCORIS GLoRA"),
        "prefix_pattern": None,
        "patterns": [r"\.a1\.weight$", r"\.a2\.weight$",
                     r"\.b1\.weight$", r"\.b2\.weight$", r"\.bm\.weight$"],
        "alpha_pattern": r"\.alpha$",
        "down_pattern": r"\.a1\.weight$",
        "up_pattern": r"\.b1\.weight$",
        "verified": "derived",
        "note": T("Needs LyCORIS support.", "LyCORIS 拡張が必要"),
    },
    {
        # lycoris/modules/ia3.py registers exactly two things: a parameter named
        # "weight" and a buffer named "on_input". There is no "ia3_weight" key -
        # an earlier version of this rule invented one. Since a bare ".weight"
        # matches almost anything, on_input is the only usable marker.
        "id": "ia3",
        "name": T("(IA)^3", "(IA)^3"),
        "prefix_pattern": None,
        "patterns": [r"\.on_input$"],
        "alpha_pattern": None,
        "down_pattern": None,
        "up_pattern": None,
        "verified": "derived",
        "note": T("Support is limited. Identified by the on_input buffer, since its "
                  "other tensor is just called \"weight\".",
                  "対応実装が限られる。もう一方のテンソル名が \"weight\" なので、"
                  "on_input バッファでのみ識別している"),
    },
    {
        # diag_oft.py and boft.py both register oft_blocks, an optional rescale
        # and an alpha buffer, and nothing else. No key starts with "boft" -
        # boft_b and boft_m are plain Python attributes, never serialised. An
        # earlier version of this rule looked for "oft_diag" and "boft_", both
        # of which were invented.
        # The two variants differ only in tensor rank:
        #   DiagOFT  oft_blocks (block_num, block_size, block_size)      -> 3 dims
        #   BOFT     oft_blocks (m, block_num, block_size, block_size)   -> 4 dims
        "id": "oft",
        "name": T("OFT / BOFT (orthogonal finetuning)", "OFT / BOFT (直交微調整)"),
        "prefix_pattern": None,
        "patterns": [r"\.oft_blocks$", r"\.rescale$"],
        "alpha_pattern": r"\.alpha$",
        "down_pattern": None,
        "up_pattern": None,
        "verified": "derived",
        "note": T("Needs LyCORIS support. A 3-dimensional oft_blocks means diagonal OFT, "
                  "4-dimensional means the butterfly variant (BOFT).",
                  "LyCORIS 拡張が必要。oft_blocks が 3 次元なら対角 OFT、"
                  "4 次元なら butterfly 版 (BOFT)"),
    },
    {
        "id": "textual_inversion",
        "name": T("Textual Inversion / embedding", "Textual Inversion / Embedding"),
        "kind": "embedding",
        "prefix_pattern": None,
        "patterns": [r"^string_to_param", r"^emb_params$", r"^clip_[lg]$", r"^clip_[lg]\."],
        "alpha_pattern": None,
        "down_pattern": None,
        "up_pattern": None,
        "verified": "measured",
        "note": T("Invoked by filename from the prompt. Goes in models/embeddings.",
                  "プロンプトにファイル名を書いて呼び出す。models/embeddings に置く"),
    },
]

# =========================================================================
# 3. Component detection - what the file contains.
# =========================================================================
COMPONENT_RULES = [
    {
        "id": "unet_ldm",
        "name": T("UNet (LDM / SAI naming)", "UNet (LDM / SAI 命名)"),
        "patterns": [r"^(input_blocks|output_blocks|middle_block)_\d+"],
        "verified": "measured",
    },
    {
        "id": "unet_diffusers",
        "name": T("UNet (diffusers naming)", "UNet (diffusers 命名)"),
        "patterns": [r"^(down_blocks|up_blocks|mid_block)_\d+",
                     r"^mid_block_(attentions|resnets)_"],
        "verified": "measured",
    },
    {
        "id": "dit_blocks",
        "name": T("DiT / transformer backbone blocks", "DiT / Transformer 本体ブロック"),
        # "layers" is safe here because a text encoder's layers arrive as
        # model_layers_N - the "model." prefix is not stripped for those.
        "patterns": [r"^(transformer_blocks|blocks|double_blocks|single_blocks"
                     r"|joint_blocks|double_stream_blocks|single_stream_blocks"
                     r"|layers|noise_refiner|context_refiner)_\d+",
                     # AuraFlow keeps its stacks under a "model." prefix that is
                     # deliberately not stripped
                     r"^model_(double|single)_layers_\d+"],
        "verified": "measured",
    },
    {
        "id": "text_encoder_clip",
        "name": T("Text encoder (CLIP family)", "Text Encoder (CLIP 系)"),
        "patterns": [r"text_model_encoder_layers_\d+", r"^\d+_transformer_text_model_"],
        "verified": "measured",
    },
    {
        "id": "text_encoder_llm",
        "name": T("Text encoder (LLM family: Qwen / T5 / Gemma ...)",
                  "Text Encoder (LLM 系: Qwen / T5 / Gemma など)"),
        "patterns": [r"^model_layers_\d+_(self_attn|mlp)_", r"^encoder_block_\d+_layer_"],
        "verified": "measured",
    },
    {
        # A normalised key has ".weight" stripped, so "quant_conv.weight" becomes
        # "quant_conv" - the pattern must not require a trailing "_".
        "id": "vae_sd",
        "name": T("VAE (SD-family 2D)", "VAE (SD 系 2D)"),
        "patterns": [
            r"^(encoder|decoder)_(down|up)_\d+_block_\d+",
            r"^(encoder|decoder)_(down_blocks|up_blocks)_\d+_resnets_",
            r"^(encoder|decoder)_mid_block_(resnets|attentions)_",
            r"^(quant_conv|post_quant_conv)$",
        ],
        "verified": "measured",
    },
    {
        "id": "vae_3d",
        "name": T("VAE (video / 3D)", "VAE (動画 / 3D 系)"),
        "patterns": [r"^(encoder_downsamples|decoder_upsamples)_\d+"],
        "verified": "measured",
    },
    {
        # Used when only the key prefix says a text encoder is present, with no
        # further clue as to which family it belongs to.
        "id": "text_encoder_generic",
        "name": T("Text encoder", "Text Encoder"),
        "patterns": [],
        "verified": "measured",
    },
    {
        "id": "controlnet",
        "name": T("ControlNet", "ControlNet"),
        "patterns": [
            r"^controlnet_(blocks|cond_embedding|down_blocks|mid_block)",
            r"^input_hint_block_", r"^zero_convs_\d+",
        ],
        "verified": "measured",
    },
]

# =========================================================================
# 4. Cross-attention context dimension -> base model.
#    This is the text encoder's output width. A LoRA or UNet simply cannot be
#    wired to a text encoder of a different width, so a match is decisive.
#    Measured: SD1.5 attn2.to_v = [640, 768]; SDXL attn2.to_v = [1280, 2048].
#    SD2.x at 1024 is derived (OpenCLIP-H output width), not measured.
# =========================================================================
CONTEXT_DIM_TO_BASE = {
    768: (T("SD1.x (Stable Diffusion 1.4 / 1.5)", "SD1.x (Stable Diffusion 1.4 / 1.5 系)"), "measured"),
    1024: (T("SD2.x (Stable Diffusion 2.0 / 2.1)", "SD2.x (Stable Diffusion 2.0 / 2.1 系)"), "derived"),
    2048: (T("SDXL family", "SDXL 系"), "measured"),
}

CROSS_ATTN_PATTERNS = [
    r"attn2_to_[kv]$",
    r"attn2_to_[kv]_lora_down_weight$",
    r"attn2_to_[kv]_lora_A_weight$",
    r"attn2_to_[kv]_weight$",
]

# =========================================================================
# 5. Architectures.
#    signals:      (pattern, weight, why-this-is-evidence)
#    context_dims: matching cross-attention width adds a decisive score
#    hidden:       (pattern, axis, expected) - a representative tensor's width
#    require_ndim: (pattern, ndim) - if such a tensor exists its rank must match,
#                  otherwise the candidate is dropped. Used where two families
#                  share naming and differ only in tensor rank (2D vs 3D VAE).
#    for:          which file kind the rule applies to; defaults to ["diffusion"]
# =========================================================================
ARCHITECTURES = [
    {
        "id": "sd15",
        "name": T("SD1.x (Stable Diffusion 1.4 / 1.5)", "SD1.x (Stable Diffusion 1.4 / 1.5 系)"),
        "verified": "measured",
        "signals": [
            # "cond_stage_model." is stripped before matching, so this has to be
            # written against what survives: transformer.text_model.*. SDXL's
            # first encoder normalises to 0_transformer_text_model_, so the
            # anchor keeps them apart.
            (r"^transformer_text_model_", 2,
             T("a single CLIP text encoder sitting directly under the checkpoint",
               "チェックポイント直下に CLIP text encoder が 1 本だけある配置")),
            (r"^input_blocks_\d+", 1, T("LDM-named UNet", "LDM 命名の UNet")),
        ],
        "context_dims": [768],
        "veto": [r"^label_emb_", r"^conditioner_embedders_1_", r"^add_embedding_"],
        "note": T("", ""),
        "comfy_dir": "checkpoints",
    },
    {
        "id": "sdxl",
        "name": T("SDXL family (SDXL 1.0 / Illustrious / Pony / NoobAI / Animagine ...)",
                  "SDXL 系 (SDXL 1.0 / Illustrious / Pony / NoobAI / Animagine ほか)"),
        "verified": "measured",
        "signals": [
            # "conditioner.embedders." is stripped, leaving "1_model_..." for
            # the second encoder. Writing this as ^conditioner_embedders_1_
            # matched nothing at all for a long time without anyone noticing,
            # because the other signals carried the verdict.
            (r"^1_model_", 4,
             T("the second text encoder (OpenCLIP-G) that only SDXL has",
               "SDXL 特有の 2 本目 text encoder (OpenCLIP-G)")),
            (r"^label_emb_\d+", 3,
             T("SDXL's label_emb (resolution and crop conditioning)",
               "SDXL UNet の label_emb（解像度・クロップ条件付け）")),
            (r"^add_embedding_linear_\d+", 3,
             T("add_embedding of the diffusers-side SDXL UNet",
               "diffusers 版 SDXL UNet の add_embedding")),
            (r"^input_blocks_8_1_transformer_blocks_", 2,
             T("transformers sitting on the deep blocks, as SDXL does",
               "SDXL の深いブロックに transformer が乗る構成")),
        ],
        "context_dims": [2048],
        "veto": [],
        "note": T("Derivatives such as Illustrious, Pony, NoobAI and Animagine are structurally "
                  "identical to SDXL 1.0, so they cannot be told apart from the weights' shape. "
                  "Kolors also lands here: it reuses the SDXL UNet and projects its ChatGLM text "
                  "encoder down to the same 2048 width, so its UNet is indistinguishable - but it "
                  "needs that ChatGLM encoder, not CLIP. Distinguishing any of these needs "
                  "metadata or the filename.",
                  "Illustrious / Pony / NoobAI / Animagine などの派生は重みの構造が SDXL 1.0 と"
                  "完全に同一のため、構造だけでは区別できない。Kolors もここに含まれる"
                  "（SDXL の UNet を流用し、ChatGLM text encoder の出力を同じ 2048 次元に"
                  "射影しているため UNet が区別できない。ただし CLIP ではなく ChatGLM が要る）。"
                  "区別にはメタデータかファイル名が要る"),
        "comfy_dir": "checkpoints",
    },
    {
        "id": "sd3",
        "name": T("SD3 / SD3.5 (MMDiT)", "SD3 / SD3.5 (MMDiT)"),
        "verified": "measured",
        "signals": [
            (r"^joint_blocks_\d+", 4, T("MMDiT joint_blocks", "MMDiT の joint_blocks")),
            (r"^x_embedder_proj$", 1, T("MMDiT patch embedder", "MMDiT の patch embedder")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("The single-file release bundles the VAE but not the text encoders, "
                  "which have to be loaded separately.",
                  "単一ファイル版は VAE を同梱するが text encoder は含まないため、"
                  "text encoder は別途読み込む必要がある"),
        "comfy_dir": "checkpoints",
    },
    {
        # Measured in both layouts. The single-file (LDM) release uses
        # double_blocks / single_blocks; the diffusers release uses
        # transformer_blocks / single_transformer_blocks, whose attention
        # submodules are byte-for-byte the same names Qwen-Image uses. The two
        # are separated by what surrounds the attention: Flux has ff /
        # ff_context / norm1_context and a context_embedder, Qwen-Image has
        # img_mlp / txt_mlp / img_mod / txt_mod. Hence the mutual vetoes.
        "id": "flux",
        "name": T("FLUX.1 (dev / schnell family)", "FLUX.1 (dev / schnell 系)"),
        "verified": "measured",
        "signals": [
            (r"^double_blocks_\d+", 4, T("Flux double-stream blocks (single-file layout)",
                                         "Flux の double stream ブロック（単一ファイル形式）")),
            (r"^single_blocks_\d+", 4, T("Flux single-stream blocks (single-file layout)",
                                         "Flux の single stream ブロック（単一ファイル形式）")),
            (r"^single_transformer_blocks_\d+", 4,
             T("Flux single-stream blocks (diffusers layout)",
               "Flux の single stream ブロック（diffusers 形式）")),
            (r"^context_embedder$", 4,
             T("a context_embedder taking T5-XXL's 4096-wide output",
               "T5-XXL の 4096 次元出力を受ける context_embedder")),
            (r"^transformer_blocks_\d+_ff_context_net_\d+", 3,
             T("a separate feed-forward for the text stream",
               "text 側に独立した feed-forward を持つ")),
            (r"^transformer_blocks_\d+_norm1_context_linear$", 2,
             T("separate adaLN modulation for the text stream",
               "text 側に独立した adaLN 変調を持つ")),
            (r"^(img_in|txt_in|guidance_in|vector_in)(_|$)", 2,
             T("Flux input embeddings (single-file layout)",
               "Flux の入力埋め込み（単一ファイル形式）")),
        ],
        "context_dims": [],
        "hidden": [(r"^context_embedder$", 1, 4096)],
        "veto": [r"^txt_in_individual_token_refiner_",
                 r"^transformer_blocks_\d+_(txt|img)_mod_",
                 r"^transformer_blocks_\d+_(txt|img)_mlp_",
                 r"^distilled_guidance_layer_"],
        "note": T("dev and schnell share the same structure. In the single-file layout "
                  "guidance_in is the only hint (dev has it); in the diffusers layout it is "
                  "time_text_embed.guidance_embedder.",
                  "dev と schnell は構造が同一。単一ファイル形式では guidance_in の有無"
                  "（dev にある）、diffusers 形式では time_text_embed.guidance_embedder が"
                  "唯一の手がかり"),
        "comfy_dir": "diffusion_models",
    },
    {
        "id": "qwen_image",
        "name": T("Qwen-Image / Qwen-Image-Edit", "Qwen-Image / Qwen-Image-Edit"),
        "verified": "measured",
        "signals": [
            (r"^transformer_blocks_\d+_(txt|img)_mod_\d+", 4,
             T("txt_mod / img_mod of the Qwen-Image DiT",
               "Qwen-Image DiT の txt_mod / img_mod")),
            (r"^transformer_blocks_\d+_attn_(add_[kqv]_proj|to_add_out)", 3,
             T("separate attention projections for the text stream",
               "text 側 attention 射影を別に持つ構成")),
            (r"^transformer_blocks_\d+_(txt|img)_mlp_net_\d+", 2,
             T("separate MLPs for the text and image streams",
               "txt/img 別系統の MLP")),
        ],
        "context_dims": [],
        "hidden": [(r"^transformer_blocks_\d+_attn_add_k_proj", 1, 3072)],
        # Flux's diffusers layout has the same attention submodule names and the
        # same hidden width of 3072. What it does not have is img_mod / txt_mod;
        # what it does have is a context_embedder and ff_context.
        "veto": [r"^context_embedder$", r"^transformer_blocks_\d+_ff_context_",
                 r"^single_transformer_blocks_\d+"],
        "note": T("Qwen-Image and Qwen-Image-Edit share the same DiT structure and cannot be "
                  "told apart from it.",
                  "Qwen-Image と Qwen-Image-Edit は DiT 構造が同一のため、構造だけでは区別できない"),
        "comfy_dir": "diffusion_models",
    },
    {
        "id": "anima",
        "name": T("Anima (DiT with an LLM adapter)", "Anima (LLM adapter 付き DiT)"),
        "verified": "measured",
        "signals": [
            (r"^llm_adapter_blocks_\d+", 4, T("the llm_adapter unique to Anima",
                                              "Anima 特有の llm_adapter")),
            (r"^blocks_\d+_adaln_modulation_(self_attn|cross_attn|mlp)_", 3,
             T("three separate adaLN modulation paths per block",
               "adaLN 変調を 3 系統持つブロック")),
            (r"^t_embedding_norm$", 2, T("Anima's t_embedding_norm", "Anima の t_embedding_norm")),
        ],
        "context_dims": [],
        "hidden": [(r"^blocks_\d+_cross_attn_k_proj$", 1, 1024)],
        "veto": [],
        "note": T("Text conditioning expects a Qwen3-0.6B-class text encoder (hidden 1024).",
                  "テキスト条件は Qwen3-0.6B 系の text encoder（hidden 1024）を前提とする"),
        "comfy_dir": "diffusion_models",
    },
    {
        # Measured against Wan-AI/Wan2.1-T2V-1.3B. Projections are single letters
        # (q/k/v/o) and QK-Norm is named norm_q/norm_k - Anima uses k_norm, the
        # other way round, which keeps the two apart.
        "id": "wan",
        "name": T("Wan 2.x (video generation)", "Wan 2.x (動画生成)"),
        "verified": "measured",
        "signals": [
            (r"^blocks_\d+_cross_attn_norm_[kq]$", 4,
             T("Wan's cross-attention QK-Norm (norm_q / norm_k)",
               "Wan の cross-attention QK-Norm (norm_q / norm_k)")),
            (r"^(patch_embedding|time_projection|text_embedding)(_|$)", 3,
             T("Wan's input embeddings", "Wan の入力埋め込み")),
            (r"^blocks_\d+_(self_attn|cross_attn)_[kqvo]$", 3,
             T("attention projections named with single letters q/k/v/o",
               "attention 射影が q/k/v/o の単文字")),
            (r"^blocks_\d+_ffn_\d+$", 1, T("Wan's FFN", "Wan の FFN")),
        ],
        "context_dims": [],
        "veto": [r"^llm_adapter_", r"^transformer_blocks_\d+_(txt|img)_mod_"],
        "note": T("T2V vs I2V and 1.3B vs 14B can be guessed from the hidden width, "
                  "but the structure is the same.",
                  "T2V と I2V、1.3B と 14B の区別は hidden 次元で推測できるが、構造は同一"),
        "comfy_dir": "diffusion_models",
    },
    {
        # Architecturally close to Flux: double_blocks / single_blocks, img_in,
        # txt_in, time_in, vector_in, guidance_in, final_layer are all shared.
        # The token refiner under txt_in is what Flux does not have, and img_in
        # is a projection here rather than a bare linear.
        "id": "hunyuan_video",
        "name": T("HunyuanVideo", "HunyuanVideo"),
        "verified": "measured",
        "signals": [
            (r"^(double_blocks|single_blocks)_\d+", 2,
             T("the same two-stage layout Flux uses", "Flux 系と同じ二段構成")),
            (r"^txt_in_individual_token_refiner_blocks_\d+", 5,
             T("the per-token text refiner unique to HunyuanVideo",
               "HunyuanVideo 特有の token refiner")),
            (r"^txt_in_(c_embedder|t_embedder|input_embedder)_", 3,
             T("HunyuanVideo's multi-part text input stage",
               "HunyuanVideo の多段構成 text 入力")),
            (r"^img_in_proj$", 2,
             T("img_in is a projection here, not a bare linear as in Flux",
               "img_in が Flux のような素の linear ではなく projection")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("The ComfyUI repackage nests the weights under model.model., which is "
                  "stripped before matching.",
                  "ComfyUI 再配布版は重みを model.model. の下に二重に入れているが、"
                  "照合前に取り除いている"),
        "comfy_dir": "diffusion_models",
    },

    {
        # Retrained from FLUX.1 schnell, so double_blocks / single_blocks /
        # img_in / txt_in are all inherited. What is new is the distilled
        # guidance stack, which replaces Flux's guidance embedding entirely.
        # The Flux rule vetoes on it.
        "id": "chroma",
        "name": T("Chroma (FLUX.1 schnell derivative)", "Chroma (FLUX.1 schnell 派生)"),
        "verified": "measured",
        "signals": [
            (r"^distilled_guidance_layer_", 5,
             T("the distilled guidance stack that replaces Flux's guidance embedding",
               "Flux の guidance 埋め込みを置き換えた distilled guidance 層")),
            (r"^(double_blocks|single_blocks)_\d+", 2,
             T("the Flux two-stage block layout it inherits",
               "Flux から受け継いだ二段ブロック構成")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("Uses the same text encoders and VAE as FLUX.1, but the loader has to "
                  "know about the distilled guidance stack.",
                  "text encoder と VAE は FLUX.1 と同じものを使うが、"
                  "distilled guidance 層に対応したローダーが要る"),
        "comfy_dir": "diffusion_models",
    },
    {
        # Mixture-of-experts feed-forward, and a "_t" suffix marking the text
        # side of every attention projection.
        "id": "hidream",
        "name": T("HiDream-I1", "HiDream-I1"),
        "verified": "measured",
        "signals": [
            (r"^double_stream_blocks_\d+_block_ff_i_experts_\d+", 5,
             T("a mixture-of-experts feed-forward", "mixture-of-experts 型 feed-forward")),
            (r"^double_stream_blocks_\d+_block_attn1_to_[kqv]_t$", 3,
             T("attention projections duplicated for the text stream (_t suffix)",
               "text 側の attention 射影が _t 付きで別に存在する")),
            (r"^p_embedder_pooled_embedder_", 2,
             T("HiDream's pooled-embedding stage", "HiDream の pooled embedding 段")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("Needs CLIP-L, CLIP-G, T5-XXL and a Llama text encoder together.",
                  "CLIP-L / CLIP-G / T5-XXL / Llama 系 text encoder を揃える必要がある"),
        "comfy_dir": "diffusion_models",
    },
    {
        # Two parallel streams named _x (image) and _y (text) all the way down.
        "id": "mochi",
        "name": T("Mochi 1 (video)", "Mochi 1 (動画生成)"),
        "verified": "measured",
        "signals": [
            (r"^blocks_\d+_attn_qkv_[xy]$", 4,
             T("separate qkv projections for the image and text streams (_x / _y)",
               "画像側 _x と text 側 _y で qkv 射影が分かれている")),
            (r"^t5_y_embedder_", 4,
             T("the T5 cross-attention embedder unique to Mochi",
               "Mochi 特有の T5 cross-attention embedder")),
            (r"^blocks_\d+_mod_[xy]$", 2,
             T("per-stream modulation", "系統別の変調")),
            (r"^pos_frequencies$", 2,
             T("Mochi's positional frequency table", "Mochi の位置周波数テーブル")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("Conditioned on T5-XXL.", "T5-XXL で条件付けする"),
        "comfy_dir": "diffusion_models",
    },
    {
        # Single-file releases carry the VAE too, under a bare "vae." prefix,
        # so these come out as backbone + VAE rather than backbone only.
        "id": "ltx_video",
        "name": T("LTX-Video", "LTX-Video"),
        "verified": "measured",
        "signals": [
            (r"^patchify_proj$", 5,
             T("LTX-Video's patchify projection", "LTX-Video の patchify 射影")),
            (r"^(adaln_single|caption_projection)_", 3,
             T("a single shared adaLN stage and a caption projection",
               "共有の adaLN 段と caption 射影")),
            (r"^transformer_blocks_\d+_scale_shift_table$", 3,
             T("a per-block scale/shift table rather than a linear modulation",
               "ブロックごとの scale/shift テーブル")),
        ],
        "context_dims": [],
        # PixArt and SANA share the adaln_single / caption_projection stage but
        # patchify the image with pos_embed.proj instead of patchify_proj.
        "veto": [r"^pos_embed_proj$", r"^caption_norm$"],
        "note": T("Conditioned on T5-XXL. The single-file releases bundle the VAE.",
                  "T5-XXL で条件付けする。単一ファイル版は VAE を同梱している"),
        "comfy_dir": "checkpoints",
    },
    {
        # One attention stack (attn1 only) with the text stream folded into the
        # patch embedder, which is what patch_embed.text_proj is for.
        "id": "cogvideox",
        "name": T("CogVideoX", "CogVideoX"),
        "verified": "measured",
        "signals": [
            (r"^patch_embed_text_proj$", 5,
             T("a text projection built into the patch embedder",
               "patch embedder に組み込まれた text 射影")),
            (r"^transformer_blocks_\d+_norm[12]_linear$", 2,
             T("adaLN modulation per block", "ブロックごとの adaLN 変調")),
            (r"^transformer_blocks_\d+_attn1_norm_[kq]$", 2,
             T("QK-Norm on a single attention stack",
               "単一 attention に対する QK-Norm")),
        ],
        "context_dims": [],
        "veto": [r"^context_embedder$", r"^transformer_blocks_\d+_(txt|img)_mod_"],
        "note": T("Conditioned on T5-XXL.", "T5-XXL で条件付けする"),
        "comfy_dir": "diffusion_models",
    },
    {
        # Z-Image and Lumina-Image 2.0 share the refiner-stack layout almost
        # exactly. What differs is the caption side: Z-Image has cap_embedder
        # and cap_pad_token with all_-prefixed embedder and head, Lumina has a
        # combined time_caption_embed. They veto each other on those.
        "id": "z_image",
        "name": T("Z-Image", "Z-Image"),
        "verified": "measured",
        "signals": [
            (r"^(noise_refiner|context_refiner)_\d+", 4,
             T("separate refiner stacks for the noise and context sides",
               "noise 側と context 側に分かれた refiner")),
            (r"^cap_(embedder|pad_token)", 4,
             T("a caption embedder with its own pad token",
               "専用の pad token を持つ caption embedder")),
            (r"^all_(x_embedder|final_layer)_", 3,
             T("multi-resolution embedder and head", "多解像度 embedder と出力層")),
        ],
        "context_dims": [],
        "veto": [r"^time_caption_embed_"],
        "note": T("", ""),
        "comfy_dir": "diffusion_models",
    },
    {
        "id": "lumina2",
        "name": T("Lumina-Image 2.0", "Lumina-Image 2.0"),
        "verified": "measured",
        "signals": [
            (r"^time_caption_embed_(caption_embedder|timestep_embedder)_", 5,
             T("a combined timestep-and-caption embedder",
               "timestep と caption をまとめた embedder")),
            (r"^(noise_refiner|context_refiner)_\d+", 4,
             T("separate refiner stacks for the noise and context sides",
               "noise 側と context 側に分かれた refiner")),
        ],
        "context_dims": [],
        "veto": [r"^cap_(embedder|pad_token)", r"^all_x_embedder_"],
        "note": T("Z-Image uses nearly the same layout; the caption stage is what separates them.",
                  "Z-Image がほぼ同じ構成を使う。caption 段の作りが違いになる"),
        "comfy_dir": "diffusion_models",
    },
    {
        "id": "hunyuan_dit",
        "name": T("HunyuanDiT", "HunyuanDiT"),
        "verified": "measured",
        "signals": [
            (r"^text_embedding_padding$", 5,
             T("a learned padding embedding for the text side",
               "text 側の学習済みパディング埋め込み")),
            (r"^time_extra_emb_", 3,
             T("HunyuanDiT's extra timestep conditioning",
               "HunyuanDiT の追加 timestep 条件付け")),
            (r"^blocks_\d+_attn2_norm_[kq]$", 2,
             T("QK-Norm on the cross-attention", "cross-attention の QK-Norm")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("Unrelated to HunyuanVideo despite the shared name. Conditioned on a "
                  "bilingual CLIP plus mT5.",
                  "名前は似ているが HunyuanVideo とは別系統。"
                  "二言語 CLIP と mT5 で条件付けする"),
        "comfy_dir": "diffusion_models",
    },
    {
        # Stage B and Stage C share these markers; modelspec.architecture in the
        # metadata says which, and the mapper tells them apart structurally.
        "id": "stable_cascade",
        "name": T("Stable Cascade (Würstchen v3)", "Stable Cascade (Würstchen v3)"),
        "verified": "measured",
        "signals": [
            (r"^(clip_mapper|clip_img_mapper)$", 5,
             T("the CLIP mapper that conditions the cascade",
               "cascade を条件付けする CLIP mapper")),
            (r"^down_blocks_\d+_\d+_(channelwise|depthwise|mapper_sca)", 3,
             T("depthwise/channelwise blocks rather than a UNet resnet stack",
               "UNet の resnet ではなく depthwise/channelwise ブロック")),
            (r"^clf_\d+$", 2, T("the output classifier head", "出力の classifier 層")),
            (r"^(effnet_mapper|pixels_mapper)_\d+", 2,
             T("Stage B's effnet and pixel mappers",
               "Stage B の effnet / pixel mapper")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("Stage C generates the small latent and Stage B decodes it; both are "
                  "needed. clip_img_mapper means Stage C, effnet_mapper means Stage B, "
                  "and modelspec.architecture in the metadata states it outright.",
                  "Stage C が小さい潜在を生成し Stage B が復号する。両方必要。"
                  "clip_img_mapper なら Stage C、effnet_mapper なら Stage B。"
                  "メタデータの modelspec.architecture にも明記されている"),
        "comfy_dir": "checkpoints",
    },
    {
        "id": "sana",
        "name": T("SANA", "SANA"),
        "verified": "measured",
        "signals": [
            (r"^caption_norm$", 5,
             T("a normalisation on the caption embedding, which PixArt lacks",
               "PixArt にはない caption 埋め込みの正規化")),
            (r"^(adaln_single|caption_projection)_", 2,
             T("the shared adaLN and caption projection stage it inherits from PixArt",
               "PixArt 譲りの共有 adaLN と caption 射影")),
            (r"^pos_embed_proj$", 2,
             T("patch embedding via pos_embed.proj", "pos_embed.proj による patch 埋め込み")),
        ],
        "context_dims": [],
        "veto": [r"^patchify_proj$"],
        "note": T("Conditioned on a Gemma text encoder rather than T5 or CLIP.",
                  "T5 や CLIP ではなく Gemma 系 text encoder で条件付けする"),
        "comfy_dir": "diffusion_models",
    },
    {
        "id": "pixart",
        "name": T("PixArt-alpha / PixArt-Sigma", "PixArt-alpha / PixArt-Sigma"),
        "verified": "measured",
        "signals": [
            (r"^(adaln_single|caption_projection)_", 4,
             T("a single shared adaLN stage and a caption projection",
               "共有の adaLN 段と caption 射影")),
            (r"^pos_embed_proj$", 3,
             T("patch embedding via pos_embed.proj", "pos_embed.proj による patch 埋め込み")),
            (r"^scale_shift_table$", 2,
             T("a top-level scale/shift table", "トップレベルの scale/shift テーブル")),
        ],
        "context_dims": [],
        # SANA is built on the same stage but adds caption_norm; LTX-Video uses
        # patchify_proj rather than pos_embed.
        "veto": [r"^caption_norm$", r"^patchify_proj$"],
        "note": T("Conditioned on T5. alpha and Sigma share this structure.",
                  "T5 で条件付けする。alpha と Sigma は構造が同一"),
        "comfy_dir": "diffusion_models",
    },
    {
        "id": "auraflow",
        "name": T("AuraFlow", "AuraFlow"),
        "verified": "measured",
        "signals": [
            # The single-file release nests everything under "model.", which is
            # not stripped - doing so would also strip a text encoder's
            # model.layers and misclassify it. Matched here instead.
            (r"^(model_)?(double|single)_layers_\d+_attn_w[12][qkvo]$", 5,
             T("attention projections named w1q / w2k and so on",
               "w1q / w2k のような命名の attention 射影")),
            (r"^(model_)?(cond_seq_linear|init_x_linear|final_linear)$", 3,
             T("AuraFlow's input and output linears",
               "AuraFlow の入出力 linear")),
            (r"^(model_)?register_tokens$", 3,
             T("learned register tokens", "学習済みレジスタトークン")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("The single-file release bundles the VAE and the T5 text encoder as well.",
                  "単一ファイル版は VAE と T5 text encoder も同梱している"),
        "comfy_dir": "checkpoints",
    },

    # ---- VAE ------------------------------------------------------------
    # 2D and 3D VAEs share naming in the diffusers layout (both use
    # encoder.down_blocks), but differ in conv rank: 2D is [out,in,h,w],
    # 3D is [out,in,t,h,w]. require_ndim separates them cleanly.
    {
        "id": "vae_3d_16ch",
        "for": ["vae"],
        "name": T("3D VAE (video models: Wan / Qwen-Image / CogVideoX / HunyuanVideo)",
                  "3D VAE (動画系: Wan / Qwen-Image / CogVideoX / HunyuanVideo)"),
        "verified": "measured",
        "signals": [
            (r"^(encoder_downsamples|decoder_upsamples)_\d+", 4,
             T("3D VAE naming with a temporal axis", "時間軸を持つ 3D VAE の命名")),
            (r"^decoder_upsamples_\d+_time_conv$", 3,
             T("convolution along the time axis", "時間方向の畳み込み")),
            (r"^(encoder|decoder)_(down_blocks|up_blocks)_\d+_resnets_", 3,
             T("VAE block layout", "VAE のブロック構成")),
            (r"^(encoder|decoder)_\w*_resnets_\d+_norm\d+_conv_[by]_conv$", 3,
             T("spatially-conditioned normalisation, as CogVideoX uses",
               "CogVideoX 系の空間条件付き正規化")),
            (r"^(quant_conv|post_quant_conv)$", 2, T("quantisation conv", "量子化 conv")),
        ],
        "context_dims": [],
        # A 2D VAE's convolutions are [out, in, h, w]; a 3D VAE adds a time axis.
        # Whichever of these tensors a given dialect names, the rank settles it.
        "require_ndim": [(r"^(quant_conv|conv1|decoder_conv_in(_conv)?)$", 5)],
        # The latent width sits on a different axis depending on which tensor
        # names it: post_quant_conv is [latent, latent, ...] so axis 0, while
        # decoder_conv_in is [out, latent, ...] so axis 1. Folding these into one
        # pattern makes the answer depend on which key happens to be found first.
        "hidden": [(r"^(post_quant_conv|conv2)$", 0, 16),
                   (r"^decoder_conv_in_conv$", 1, 16)],
        "veto": [],
        "note": T("Several video models share this family of 3D VAE, so which one a given "
                  "file belongs to is not reliably separable from structure alone.",
                  "複数の動画モデルが同系の 3D VAE を使うため、"
                  "どれ用かを構造だけで区別するのは困難"),
        "comfy_dir": "vae",
    },
    {
        "id": "vae_sd_2d",
        "for": ["vae"],
        "name": T("VAE (SD1.x / SDXL family, 2D)", "VAE (SD1.x / SDXL 系 2D)"),
        "verified": "measured",
        "signals": [
            (r"^(encoder|decoder)_(down|up)_\d+_block_\d+", 4,
             T("LDM-named VAE blocks", "LDM 命名の VAE ブロック")),
            (r"^(encoder|decoder)_(down_blocks|up_blocks)_\d+_resnets_", 4,
             T("diffusers-named VAE blocks", "diffusers 命名の VAE ブロック")),
            (r"^(quant_conv|post_quant_conv)$", 3,
             T("the quantisation conv of an SD-family VAE", "SD 系 VAE の量子化 conv")),
        ],
        "context_dims": [],
        "require_ndim": [(r"^(quant_conv|conv1|decoder_conv_in(_conv)?)$", 4)],
        "veto": [],
        "note": T("The SD1.x and SDXL VAEs are structurally identical; only the weight values "
                  "differ (SDXL has a revision that does not break down in fp16).",
                  "SD1.x 用と SDXL 用は構造が同一。区別は重みの値でしか付かない"
                  "（SDXL には fp16 で破綻しない改良版がある）"),
        "comfy_dir": "vae",
    },

    # ---- Text encoders --------------------------------------------------
    {
        "id": "te_qwen_vl",
        "for": ["text_encoder"],
        "name": T("Qwen2.5-VL text/vision encoder", "Qwen2.5-VL 系 Text/Vision Encoder"),
        "verified": "measured",
        "signals": [
            (r"^visual_blocks_\d+", 5, T("a vision encoder is bundled in",
                                         "視覚エンコーダを内蔵している")),
            (r"^visual_patch_embed_proj$", 3, T("ViT patch embedding", "ViT の patch embedding")),
            (r"^model_layers_\d+_self_attn_[kq]_norm$", 1,
             T("Qwen-style QK-Norm", "Qwen 系の QK-Norm")),
        ],
        "context_dims": [],
        "hidden": [(r"^model_embed_tokens$", 1, 3584)],
        "veto": [],
        "note": T("Used as the text encoder for Qwen-Image / Qwen-Image-Edit (7B class).",
                  "Qwen-Image / Qwen-Image-Edit の text encoder として使う（7B 相当）"),
        "comfy_dir": "text_encoders",
    },
    {
        "id": "te_qwen_llm",
        "for": ["text_encoder"],
        "name": T("Qwen-family LLM text encoder", "Qwen 系 LLM Text Encoder"),
        "verified": "measured",
        "signals": [
            (r"^model_layers_\d+_self_attn_[kq]_norm$", 4,
             T("Qwen-style QK-Norm", "Qwen 系の QK-Norm")),
            (r"^model_layers_\d+_mlp_(gate|up|down)_proj$", 2,
             T("SwiGLU-style MLP", "SwiGLU 型 MLP")),
        ],
        "context_dims": [],
        "veto": [r"^visual_blocks_"],
        "note": T("hidden 1024 means a Qwen3-0.6B class encoder (used by Anima); "
                  "3584 means the 7B class (used by Qwen-Image).",
                  "hidden 1024 なら Qwen3-0.6B 相当（Anima 用）、3584 なら 7B 相当（Qwen-Image 用）"),
        "comfy_dir": "text_encoders",
    },
    {
        "id": "te_clip",
        "for": ["text_encoder"],
        "name": T("CLIP text encoder", "CLIP Text Encoder"),
        "verified": "measured",
        "signals": [
            (r"text_model_encoder_layers_\d+", 5,
             T("CLIP text encoder layer naming", "CLIP text encoder のレイヤ命名")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("Width 768 is CLIP-L (the first encoder of SD1.x and SDXL); "
                  "1280 is OpenCLIP-G (SDXL's second encoder).",
                  "出力次元 768 なら CLIP-L (SD1.x / SDXL の 1 本目)、"
                  "1280 なら OpenCLIP-G (SDXL の 2 本目)"),
        "comfy_dir": "text_encoders",
    },
    # ---- Textual Inversion --------------------------------------------------
    # An embedding is just the text encoder's output width, so the tensor shape
    # names the base directly. SDXL needs one vector per encoder, hence two
    # tensors; SD1.x and SD2.x carry a single emb_params.
    {
        "id": "ti_sdxl",
        "for": ["embedding"],
        "name": T("Textual Inversion for SDXL", "Textual Inversion (SDXL 用)"),
        "verified": "measured",
        "signals": [
            (r"^clip_g$", 5, T("an OpenCLIP-G side, which only SDXL has",
                               "SDXL にしかない OpenCLIP-G 側を持つ")),
            (r"^clip_l$", 2, T("a CLIP-L side", "CLIP-L 側を持つ")),
        ],
        "context_dims": [],
        "hidden": [(r"^clip_g$", 1, 1280)],
        "veto": [],
        "note": T("SDXL has two text encoders, so an embedding for it carries one "
                  "vector per encoder.",
                  "SDXL は text encoder が 2 本あるため、embedding も 2 本ぶんを持つ"),
        "comfy_dir": "embeddings",
    },
    {
        "id": "ti_sd15",
        "for": ["embedding"],
        "name": T("Textual Inversion for SD1.x", "Textual Inversion (SD1.x 用)"),
        "verified": "measured",
        "signals": [
            (r"^(emb_params|string_to_param)", 4,
             T("a single embedding tensor", "単一の embedding テンソル")),
        ],
        "context_dims": [],
        "require_dim": [(r"^emb_params$", 1, 768)],
        "veto": [r"^clip_g$"],
        "note": T("The vector count (the first axis) is how many prompt tokens it occupies.",
                  "第 1 軸の数はプロンプト上で占めるトークン数"),
        "comfy_dir": "embeddings",
    },
    {
        "id": "ti_sd2",
        "for": ["embedding"],
        "name": T("Textual Inversion for SD2.x", "Textual Inversion (SD2.x 用)"),
        "verified": "derived",
        "signals": [
            (r"^(emb_params|string_to_param)", 4,
             T("a single embedding tensor", "単一の embedding テンソル")),
        ],
        "context_dims": [],
        "require_dim": [(r"^emb_params$", 1, 1024)],
        "veto": [r"^clip_g$"],
        "note": T("Inferred from OpenCLIP-H's 1024-wide output; no real file checked.",
                  "OpenCLIP-H の出力次元 1024 から導出。実ファイル未確認"),
        "comfy_dir": "embeddings",
    },

    {
        "id": "te_t5",
        "for": ["text_encoder"],
        "name": T("T5-family text encoder", "T5 系 Text Encoder"),
        "verified": "measured",
        "signals": [
            (r"^encoder_block_\d+_layer_\d+_(SelfAttention|DenseReluDense)_", 5,
             T("T5 encoder naming", "T5 の encoder 命名")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("T5-XXL as used by FLUX.1 and SD3.",
                  "FLUX.1 / SD3 が使う T5-XXL など"),
        "comfy_dir": "text_encoders",
    },
]

# =========================================================================
# 5b. Quantisation formats, checked in order - the first match wins.
#     key_patterns match raw tensor keys; dtypes match the dtype names present.
# =========================================================================
QUANT_RULES = [
    {
        # Nunchaku / SVDQuant: INT4 weights (packed into I32) plus a low-rank
        # correction. Note the lora_down / lora_up keys - they are part of the
        # quantisation scheme, not a LoRA. They lack the ".weight" suffix that
        # the LoRA dialects require, so they do not collide.
        "id": "quant_svdquant",
        "name": T("SVDQuant / Nunchaku (INT4 weights with a low-rank correction)",
                  "SVDQuant / Nunchaku (INT4 + 低ランク補正)"),
        "key_patterns": [r"\.qweight$", r"\.wscales$"],
        "dtypes": [],
        "verified": "measured",
    },
    {
        "id": "quant_fp8_scaled",
        "name": T("fp8 scaled (ComfyUI layout, carries scale_weight)",
                  "fp8 scaled (ComfyUI 形式。scale_weight を伴う)"),
        "key_patterns": [r"\.scale_weight$"],
        "dtypes": [],
        "verified": "measured",
    },
    {
        "id": "quant_fp8",
        "name": T("fp8 (no scale correction)", "fp8 (スケール補正なし)"),
        "key_patterns": [],
        "dtypes": ["F8_E4M3", "F8_E5M2", "F8_E8M0", "F8_E4M3FNUZ", "F8_E5M2FNUZ"],
        "verified": "measured",
    },
    {
        "id": "quant_int8",
        "name": T("possibly 8-bit quantized (contains U8 / I8 tensors)",
                  "8bit 量子化の可能性（U8 / I8 テンソルを含む）"),
        "key_patterns": [],
        "dtypes": ["U8", "I8"],
        "verified": "measured",
    },
]

# =========================================================================
# 6. Where each kind of file goes, and how it is loaded.
#    ComfyUI maps the legacy names "unet" -> "diffusion_models" and
#    "clip" -> "text_encoders", so either folder works.
# =========================================================================
PLACEMENT = {
    "checkpoint": ("models/checkpoints",
                   T("Load Checkpoint node, or A1111's models/Stable-diffusion",
                     "Load Checkpoint ノード / A1111 の models/Stable-diffusion")),
    "unet_only": ("models/diffusion_models",
                  T("Load Diffusion Model node. A VAE and a text encoder are needed separately",
                    "Load Diffusion Model ノード。VAE と text encoder は別途必要")),
    "backbone_vae": ("models/checkpoints",
                     T("Load Checkpoint node, then feed the text encoders in separately "
                       "(TripleCLIPLoader for SD3.5, DualCLIPLoader for Flux)",
                       "Load Checkpoint ノード。text encoder は別途読み込む"
                       "（SD3.5 なら TripleCLIPLoader、Flux なら DualCLIPLoader）")),
    "lora": ("models/loras",
             T("LoraLoader node. The base model has to match",
               "LoraLoader ノード。ベースモデルを合わせること")),
    "vae": ("models/vae", T("Load VAE node", "Load VAE ノード")),
    "text_encoder": ("models/text_encoders",
                     T("CLIPLoader / DualCLIPLoader node", "CLIPLoader / DualCLIPLoader ノード")),
    "controlnet": ("models/controlnet", T("ControlNetLoader node", "ControlNetLoader ノード")),
    "embedding": ("models/embeddings",
                  T("Call it from the prompt as embedding:filename",
                    "プロンプト中に embedding:ファイル名 で呼び出す")),
    "unknown": (T("(not identified)", "(判別不能)"), T("", "")),
}

# =========================================================================
# 7. Metadata keys worth surfacing first.
#    Keys marked (measured) were seen in real files.
# =========================================================================
# Metadata display categories, in the order they appear. The HTML report lets
# the reader switch these on and off; everything a file carries is embedded
# either way, so nothing is lost by unticking one.
META_CATEGORIES = [
    ("identity", T("identity", "素性・名前")),
    ("origin", T("origin", "作者・ライセンス")),
    ("lineage", T("lineage", "マージ元・親")),
    ("training", T("training", "学習設定")),
    ("image", T("preview", "見本画像")),
    ("software", T("software and route", "ソフトと経路")),
    ("hash", T("hashes", "ハッシュ")),
    ("other", T("other", "その他")),
]

# What each metadata key means, and where reading it plainly goes wrong.
#
#   key      the metadata key as written in the file
#   cat      one of META_CATEGORIES
#   label    what to call it
#   explain  what the value means. ABSENT means the value speaks for itself -
#            registered deliberately, so that a key missing from this table can
#            be reported as "not understood" instead of silently blank
#   caveat   how reading it plainly misleads. Shown even where the explanation
#            is hidden, because whoever is about to be misled will not open it
#   display  "text" (default), "image", or "json"
#   bulky    printed as a length rather than in full unless --meta is given
#
# The order here is the order the report shows them in.
META_GUIDE = [
    # -- identity: what this file calls itself -----------------------------
    {"key": "modelspec.architecture", "cat": "identity",
     "label": T("declared architecture", "宣言アーキテクチャ"),
     "explain": T("the architecture the producing tool declared. The report "
                  "checks it against what the weights actually look like",
                  "作成ツールが宣言したアーキテクチャ。重みから読んだ判定と照合される")},
    {"key": "modelspec.title", "cat": "identity", "label": T("title", "タイトル"),
     "explain": T("what the producing tool called it. Every OneTrainer output "
                  "measured here says \"Stable Diffusion XL 1.0 Base LoRA\", "
                  "whatever the file actually is",
                  "作成ツールが付けた名前。実測した OneTrainer 出力は"
                  "中身によらずすべて \"Stable Diffusion XL 1.0 Base LoRA\""),
     "caveat": T("may name the base model rather than this file",
                 "このファイルではなくベースモデルを指している場合がある")},
    {"key": "name", "cat": "identity", "label": T("name", "名前")},
    {"key": "version", "cat": "identity", "label": T("version", "バージョン")},
    {"key": "ss_output_name", "cat": "identity",
     "label": T("output name", "出力名"),
     "explain": T("the file name the training run was told to write",
                  "学習時に指定された出力ファイル名")},
    {"key": "modelspec.sai_model_spec", "cat": "identity",
     "label": T("modelspec version", "modelspec の版"),
     "explain": T("the version of the metadata format itself",
                  "メタデータ形式そのものの版番号"),
     "caveat": T("not the version of the model", "モデルの版ではない")},
    {"key": "format", "cat": "identity", "label": T("tensor format", "テンソル形式"),
     "explain": T("\"pt\" means the tensors came from PyTorch",
                  "\"pt\" なら元は PyTorch のテンソル")},

    # -- origin: who and when ----------------------------------------------
    {"key": "modelspec.author", "cat": "origin", "label": T("author", "作者"),
     "explain": T("who the producing tool credited. All 13 files measured here "
                  "credit \"StabilityAI\" - the base model's author, not the "
                  "person who trained them",
                  "作成ツールが記録した作者。実測 13 件はすべて "
                  "\"StabilityAI\"、つまり学習した人ではなくベースモデルの作者"),
     "caveat": T("not necessarily whoever made this file",
                 "このファイルを作った人とは限らない")},
    {"key": "modelspec.date", "cat": "origin", "label": T("created", "作成日")},
    {"key": "modelspec.license", "cat": "origin", "label": T("license", "ライセンス"),
     "explain": T("written into the same modelspec block as the author and "
                  "title, and copied along with them",
                  "作者やタイトルと同じ modelspec に書かれ、まとめて写される"),
     "caveat": T("may be the base model's licence, not this file's",
                 "ベースモデルのライセンスである場合がある")},
    {"key": "modelspec.description", "cat": "origin",
     "label": T("description", "説明")},

    # -- lineage: what it came from ----------------------------------------
    {"key": "modelspec.merged_from", "cat": "lineage",
     "label": T("merged from", "マージ元"),
     "explain": T("the models this was merged from. Ratios are not recorded "
                  "here", "マージ元のモデル名。比率は記録されない")},
    {"key": "merged_loras", "cat": "lineage",
     "label": T("merged LoRAs", "マージした LoRA"),
     "explain": T("pairs up with merged_strengths, in the same order",
                  "merged_strengths と同じ順で対応する")},
    {"key": "merged_strengths", "cat": "lineage",
     "label": T("merge strengths", "マージ強度"),
     "explain": T("pairs up with merged_loras, in the same order",
                  "merged_loras と同じ順で対応する")},
    {"key": "merge_type", "cat": "lineage", "label": T("merge type", "マージ方式"),
     "explain": T("how the parts were combined; the strengths mean different "
                  "things per method",
                  "どう合成したか。方式によって強度の意味が変わる")},
    {"key": "merge_density", "cat": "lineage",
     "label": T("merge density", "マージ密度"),
     "explain": T("a coefficient whose meaning depends on the merge type",
                  "マージ方式に依存する係数")},
    {"key": "ss_base_model_version", "cat": "lineage",
     "label": T("training base", "学習時のベース"),
     "explain": T("the base the training run was pointed at. Trainers write "
                  "truncated identifiers such as \"sdxl_\"",
                  "学習に使ったベースの識別子。\"sdxl_\" のように"
                  "途中で切れた値もそのまま入る")},
    {"key": "ss_sd_model_name", "cat": "lineage",
     "label": T("base model name", "学習時のベースモデル名")},
    {"key": "workflow", "cat": "lineage",
     "label": T("ComfyUI workflow", "ComfyUI ワークフロー"),
     "explain": T("the node graph ComfyUI embeds when it saves a model. It "
                  "names the models that were loaded and the merge ratio, so "
                  "the parents can be read off it",
                  "ComfyUI が保存時に埋め込むノードグラフ。読み込んだモデル名と"
                  "マージ比率が入っており、親を辿れる"),
     "display": "json", "bulky": True},
    {"key": "prompt", "cat": "lineage",
     "label": T("ComfyUI graph", "ComfyUI 実行グラフ"),
     "explain": T("the same graph in the form ComfyUI executes: terser, with "
                  "each node's inputs spelled out",
                  "同じグラフの実行用の形。より簡潔で、ノードの入力値がそのまま入る"),
     "display": "json", "bulky": True},

    # -- training ----------------------------------------------------------
    {"key": "ss_network_module", "cat": "training",
     "label": T("network type", "ネットワーク種別"),
     "explain": T("networks.lora is an ordinary LoRA; lycoris.kohya is one of "
                  "the LyCORIS variants",
                  "networks.lora なら通常の LoRA、lycoris.kohya なら LyCORIS 系")},
    {"key": "ss_network_dim", "cat": "training",
     "label": T("rank (network_dim)", "rank (network_dim)")},
    {"key": "ss_network_alpha", "cat": "training", "label": T("alpha", "alpha")},
    {"key": "ss_network_args", "cat": "training",
     "label": T("network args", "ネットワーク引数")},
    {"key": "ss_resolution", "cat": "training",
     "label": T("training resolution", "学習解像度")},
    {"key": "ss_clip_skip", "cat": "training", "label": T("clip skip", "clip skip"),
     "explain": T("how many layers to drop from the end of the text encoder",
                  "テキストエンコーダの後ろから何層を捨てるか")},
    {"key": "ss_num_train_images", "cat": "training",
     "label": T("training images", "学習画像数")},
    {"key": "ss_num_epochs", "cat": "training", "label": T("epochs", "エポック数")},
    {"key": "ss_learning_rate", "cat": "training",
     "label": T("learning rate", "学習率")},
    {"key": "training_info", "cat": "training",
     "label": T("training progress", "学習進捗")},
    {"key": "ss_tag_frequency", "cat": "training",
     "label": T("tag frequency", "タグ頻度"),
     "explain": T("the tags on the training images and how often each "
                  "appeared. The most frequent are the trigger word candidates",
                  "学習画像に付いていたタグと出現数。上位がトリガー語の候補になる"),
     "bulky": True},
    {"key": "ss_datasets", "cat": "training",
     "label": T("training datasets", "学習データセット"),
     "explain": T("the training configuration as JSON. It can contain the "
                  "image directories from the machine that trained it",
                  "学習設定の JSON。学習した機械の画像ディレクトリを含むことがある"),
     "display": "json", "bulky": True},
    {"key": "ss_dataset_dirs", "cat": "training",
     "label": T("dataset directories", "学習ディレクトリ"),
     "explain": T("image counts per directory, directory names included",
                  "ディレクトリごとの画像数。ディレクトリ名を含む"),
     "bulky": True},
    {"key": "ss_bucket_info", "cat": "training",
     "label": T("bucket info", "バケット情報"),
     "explain": T("how many images fell into each training resolution",
                  "学習解像度ごとの画像数"),
     "bulky": True},
    {"key": "modelspec.resolution", "cat": "training",
     "label": T("resolution", "解像度")},
    {"key": "modelspec.prediction_type", "cat": "training",
     "label": T("prediction type", "予測タイプ"),
     "explain": T("epsilon predicts the noise, v_prediction the velocity. The "
                  "sampler has to be set to match",
                  "epsilon はノイズ予測、v_prediction は速度予測。"
                  "サンプラー側の設定と揃える必要がある")},

    # -- preview image -----------------------------------------------------
    {"key": "modelspec.thumbnail", "cat": "image",
     "label": T("preview", "見本画像"), "display": "image", "bulky": True},

    # -- software and route ------------------------------------------------
    {"key": "software", "cat": "software",
     "label": T("producing software", "作成ソフトウェア")},
    {"key": "modelspec.implementation", "cat": "software",
     "label": T("implementation", "実装"),
     "explain": T("the repository of the implementation this format targets",
                  "この形式が想定している実装のリポジトリ"),
     "caveat": T("not where this file was distributed from",
                 "このファイルの配布元ではない")},
    {"key": "ot_branch", "cat": "software",
     "label": T("OneTrainer branch", "OneTrainer ブランチ"),
     "explain": T("the branch of OneTrainer that wrote the file",
                  "書き出した OneTrainer のブランチ")},
    {"key": "ot_revision", "cat": "software",
     "label": T("OneTrainer revision", "OneTrainer リビジョン"),
     "explain": T("the commit of OneTrainer that wrote the file",
                  "書き出した OneTrainer のコミット")},

    # -- hashes ------------------------------------------------------------
    {"key": "modelspec.hash_sha256", "cat": "hash",
     "label": T("SHA256", "SHA256"),
     "explain": T("SHA256 of the weights; use it to confirm two files are the "
                  "same", "重みの SHA256。同一ファイルかの照合に使える")},
    {"key": "sshs_model_hash", "cat": "hash",
     "label": T("model hash", "モデルハッシュ"),
     "explain": T("kohya's own hash, unrelated to SHA256",
                  "kohya 系が書く独自ハッシュ。SHA256 とは別物")},
    {"key": "sshs_legacy_hash", "cat": "hash",
     "label": T("legacy model hash", "旧モデルハッシュ"),
     "explain": T("the older form of kohya's hash",
                  "kohya の旧方式のハッシュ")},
]

# Suffixes that mark a value as naming a model file. Used to read the parents
# out of an embedded ComfyUI graph by what the values look like, rather than by
# a table of node classes and their input names - anything beyond the one graph
# measured here would be a guess, and this project has written imaginary keys
# before. ".safetensors" is measured; the rest follow from the loaders ComfyUI
# ships and are derived, not confirmed against a file.
MODEL_FILE_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf")

# Derived views, so the definition above stays the only place a key is named.
META_HIGHLIGHT = [(e["key"], e["label"]) for e in META_GUIDE]
META_BULKY = [e["key"] for e in META_GUIDE if e.get("bulky")]
META_BY_KEY = {e["key"]: e for e in META_GUIDE}

# Measured values: "stable-diffusion-xl-v1-base", "stable-diffusion-xl-v1-base/lora"
MODELSPEC_ARCH_MAP = {
    "stable-diffusion-v1": "SD1.x",
    "stable-diffusion-xl-v1-base": "SDXL",
    "stable-diffusion-xl-v1-refiner": "SDXL Refiner",
    "stable-diffusion-v3": "SD3",
    "flux-1-dev": "FLUX.1 dev",
    "flux-1-schnell": "FLUX.1 schnell",
}
