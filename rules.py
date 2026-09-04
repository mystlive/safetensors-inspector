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
    ("diffusion_model.", "unet"),
    ("transformer.", "unet"),
    ("unet.", "unet"),
    ("net.", "unet"),
    ("cond_stage_model.", "text_encoder"),
    ("conditioner.embedders.", "text_encoder"),
    ("first_stage_model.", "vae"),
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
        "patterns": [r"^(transformer_blocks|blocks|double_blocks|single_blocks|joint_blocks)_\d+"],
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
            (r"^cond_stage_model", 2, T("SD1.x text encoder placement",
                                        "SD1.x の text encoder 配置")),
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
            (r"^conditioner_embedders_1_", 4,
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
                  "Distinguishing them needs metadata or the filename.",
                  "Illustrious / Pony / NoobAI / Animagine などの派生は重みの構造が SDXL 1.0 と"
                  "完全に同一のため、構造だけでは区別できない。区別にはメタデータかファイル名が要る"),
        "comfy_dir": "checkpoints",
    },
    {
        "id": "sd3",
        "name": T("SD3 / SD3.5 (MMDiT)", "SD3 / SD3.5 (MMDiT)"),
        "verified": "unverified",
        "signals": [
            (r"^joint_blocks_\d+", 4, T("MMDiT joint_blocks", "MMDiT の joint_blocks")),
            (r"^x_embedder_proj$", 1, T("MMDiT patch embedder", "MMDiT の patch embedder")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("Not verified against a real file (the repository is gated).",
                  "実ファイル未確認（リポジトリが gated のため）"),
        "comfy_dir": "checkpoints",
    },
    {
        "id": "flux",
        "name": T("FLUX.1 (dev / schnell family)", "FLUX.1 (dev / schnell 系)"),
        "verified": "unverified",
        "signals": [
            (r"^double_blocks_\d+", 4, T("Flux double-stream blocks", "Flux の double stream ブロック")),
            (r"^single_blocks_\d+", 4, T("Flux single-stream blocks", "Flux の single stream ブロック")),
            (r"^(img_in|txt_in|guidance_in|vector_in)(_|$)", 2,
             T("Flux input embeddings", "Flux の入力埋め込み")),
        ],
        "context_dims": [],
        "veto": [r"^txt_in_individual_token_refiner_"],
        "note": T("Not verified against a real file (the repository is gated). "
                  "dev and schnell share the same structure; the presence of guidance_in is the only hint.",
                  "実ファイル未確認（リポジトリが gated のため）。"
                  "dev と schnell は構造が同一で、guidance_in の有無が唯一の手がかり"),
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
        "veto": [],
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
        "id": "hunyuan_video",
        "name": T("HunyuanVideo", "HunyuanVideo"),
        "verified": "unverified",
        "signals": [
            (r"^(double_blocks|single_blocks)_\d+", 2,
             T("the same two-stage layout Flux uses", "Flux 系と同じ二段構成")),
            (r"^txt_in_individual_token_refiner_", 4,
             T("the token refiner unique to HunyuanVideo",
               "HunyuanVideo 特有の token refiner")),
        ],
        "context_dims": [],
        "veto": [],
        "note": T("Not verified against a real file.", "実ファイル未確認"),
        "comfy_dir": "diffusion_models",
    },

    # ---- VAE ------------------------------------------------------------
    # 2D and 3D VAEs share naming in the diffusers layout (both use
    # encoder.down_blocks), but differ in conv rank: 2D is [out,in,h,w],
    # 3D is [out,in,t,h,w]. require_ndim separates them cleanly.
    {
        "id": "vae_3d_16ch",
        "for": ["vae"],
        "name": T("3D VAE (Wan 2.x / Qwen-Image family)", "3D VAE (Wan 2.x / Qwen-Image 系)"),
        "verified": "measured",
        "signals": [
            (r"^(encoder_downsamples|decoder_upsamples)_\d+", 4,
             T("3D VAE naming with a temporal axis", "時間軸を持つ 3D VAE の命名")),
            (r"^decoder_upsamples_\d+_time_conv$", 3,
             T("convolution along the time axis", "時間方向の畳み込み")),
            (r"^(encoder|decoder)_(down_blocks|up_blocks)_\d+_resnets_", 3,
             T("VAE block layout", "VAE のブロック構成")),
            (r"^(quant_conv|post_quant_conv)$", 2, T("quantisation conv", "量子化 conv")),
        ],
        "context_dims": [],
        "require_ndim": [(r"^(quant_conv|conv1)$", 5)],
        "hidden": [(r"^(post_quant_conv|conv2)$", 0, 16)],
        "veto": [],
        "note": T("Wan 2.x and Qwen-Image use the same family of 3D VAE, so telling those two "
                  "apart from structure alone is not reliable.",
                  "Wan 2.x と Qwen-Image は同系の 3D VAE を使うため、この 2 つの区別は構造からは困難"),
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
        "require_ndim": [(r"^(quant_conv|conv1)$", 4)],
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
META_HIGHLIGHT = [
    ("ss_base_model_version", T("training base", "学習時のベース")),           # measured
    ("ss_sd_model_name", T("base model name", "学習時のベースモデル名")),
    ("ss_output_name", T("output name", "出力名")),                              # measured
    ("ss_network_module", T("network type", "ネットワーク種別")),
    ("ss_network_dim", T("rank (network_dim)", "rank (network_dim)")),
    ("ss_network_alpha", T("alpha", "alpha")),
    ("ss_network_args", T("network args", "ネットワーク引数")),
    ("ss_resolution", T("training resolution", "学習解像度")),
    ("ss_clip_skip", T("clip skip", "clip skip")),
    ("ss_num_train_images", T("training images", "学習画像数")),
    ("ss_num_epochs", T("epochs", "エポック数")),
    ("ss_learning_rate", T("learning rate", "学習率")),
    ("modelspec.architecture", T("declared architecture", "宣言アーキテクチャ")),  # measured
    ("modelspec.title", T("title", "タイトル")),                                  # measured
    ("modelspec.resolution", T("resolution", "解像度")),                          # measured
    ("modelspec.prediction_type", T("prediction type", "予測タイプ")),            # measured
    ("modelspec.implementation", T("implementation", "実装")),                    # measured
    ("modelspec.merged_from", T("merged from", "マージ元")),                      # measured
    ("modelspec.date", T("created", "作成日")),                                   # measured
    ("modelspec.author", T("author", "作者")),                                    # measured
    ("modelspec.description", T("description", "説明")),                          # measured
    ("software", T("producing software", "作成ソフトウェア")),                     # measured
    ("training_info", T("training progress", "学習進捗")),                        # measured
    ("name", T("name", "名前")),                                                  # measured
    ("version", T("version", "バージョン")),                                      # measured
    ("ot_branch", T("OneTrainer branch", "OneTrainer ブランチ")),                 # measured
    ("ot_revision", T("OneTrainer revision", "OneTrainer リビジョン")),           # measured
]

# Metadata values that are too bulky to print raw.
META_BULKY = ["modelspec.thumbnail", "ss_tag_frequency", "workflow", "prompt",
              "ss_dataset_dirs", "ss_bucket_info", "ss_datasets"]

# Measured values: "stable-diffusion-xl-v1-base", "stable-diffusion-xl-v1-base/lora"
MODELSPEC_ARCH_MAP = {
    "stable-diffusion-v1": "SD1.x",
    "stable-diffusion-xl-v1-base": "SDXL",
    "stable-diffusion-xl-v1-refiner": "SDXL Refiner",
    "stable-diffusion-v3": "SD3",
    "flux-1-dev": "FLUX.1 dev",
    "flux-1-schnell": "FLUX.1 schnell",
}
