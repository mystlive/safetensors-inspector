# -*- coding: utf-8 -*-
"""
Verify the detection rules against public models, using headers only.

No weights are downloaded: an HTTP Range request fetches the header and nothing
else. Headers are cached as JSON under tools/headers/, so repeat runs are offline.

Only the Hugging Face repository id and filename are recorded - never a local path.

Gated repositories (FLUX, SD 3.5) return 401 unless you have accepted their
licence on Hugging Face yourself and set HF_TOKEN.

    python tools/verify_rules.py            # everything
    python tools/verify_rules.py sdxl       # only ids containing "sdxl"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rules              # noqa: E402
import stinspect          # noqa: E402
from fetch_header import fetch_header  # noqa: E402

HEADER_DIR = Path(__file__).resolve().parent / "headers"

# 検証対象。gated かどうかと license は Hugging Face の API で確認した値。
# gated=True のものは、利用者本人がライセンスに同意し HF_TOKEN を設定した場合のみ取得できる。
TARGETS = [
    # --- SD1.x ---
    dict(id="sd15-single", genre="Checkpoint (SD1.5, single-file)",
         repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
         file="v1-5-pruned-emaonly.safetensors",
         license="CreativeML Open RAIL-M", gated=False,
         expect_kind="checkpoint", expect_arch="sd15"),
    dict(id="sd15-unet", genre="UNet only (SD1.5, diffusers layout)",
         repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
         file="unet/diffusion_pytorch_model.fp16.safetensors",
         license="CreativeML Open RAIL-M", gated=False,
         expect_kind="unet_only", expect_arch="sd15"),
    dict(id="sd15-te", genre="Text Encoder (CLIP-L)",
         repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
         file="text_encoder/model.fp16.safetensors",
         license="CreativeML Open RAIL-M", gated=False,
         expect_kind="text_encoder", expect_arch="te_clip"),
    dict(id="sd15-vae", genre="VAE (SD 2D)",
         repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
         file="vae/diffusion_pytorch_model.fp16.safetensors",
         license="CreativeML Open RAIL-M", gated=False,
         expect_kind="vae", expect_arch="vae_sd_2d"),

    # --- SDXL ---
    dict(id="sdxl-single", genre="Checkpoint (SDXL, single-file)",
         repo="stabilityai/stable-diffusion-xl-base-1.0",
         file="sd_xl_base_1.0.safetensors",
         license="OpenRAIL++", gated=False,
         expect_kind="checkpoint", expect_arch="sdxl"),
    dict(id="sdxl-lora-kohya", genre="LoRA (SDXL, official offset example)",
         repo="stabilityai/stable-diffusion-xl-base-1.0",
         file="sd_xl_offset_example-lora_1.0.safetensors",
         license="OpenRAIL++", gated=False,
         expect_kind="lora", expect_arch="sdxl"),
    dict(id="sdxl-te2", genre="Text Encoder (OpenCLIP-G)",
         repo="stabilityai/stable-diffusion-xl-base-1.0",
         file="text_encoder_2/model.fp16.safetensors",
         license="OpenRAIL++", gated=False,
         expect_kind="text_encoder", expect_arch="te_clip"),
    dict(id="sdxl-vae", genre="VAE (SDXL, standalone)",
         repo="stabilityai/sdxl-vae",
         file="diffusion_pytorch_model.safetensors",
         license="MIT", gated=False,
         expect_kind="vae", expect_arch="vae_sd_2d"),
    dict(id="sdxl-lora-peft", genre="LoRA (SDXL, PEFT layout)",
         repo="latent-consistency/lcm-lora-sdxl",
         file="pytorch_lora_weights.safetensors",
         license="OpenRAIL++", gated=False,
         expect_kind="lora", expect_arch="sdxl"),
    dict(id="sdxl-controlnet", genre="ControlNet (SDXL)",
         repo="diffusers/controlnet-canny-sdxl-1.0",
         file="diffusion_pytorch_model.fp16.safetensors",
         license="OpenRAIL++", gated=False,
         expect_kind="controlnet", expect_arch="sdxl"),

    # --- DiT 系 ---
    dict(id="qwen-image-dit", genre="DiT (Qwen-Image, sharded)",
         repo="Qwen/Qwen-Image",
         file="transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="unet_only", expect_arch="qwen_image"),
    dict(id="qwen-image-vae", genre="VAE (Qwen-Image 3D)",
         repo="Qwen/Qwen-Image",
         file="vae/diffusion_pytorch_model.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="vae", expect_arch="vae_3d_16ch"),
    dict(id="hunyuan-t2v", genre="Video DiT (HunyuanVideo T2V)",
         repo="Comfy-Org/HunyuanVideo_repackaged",
         file="split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors",
         license="Tencent Hunyuan Community", gated=False,
         expect_kind="unet_only", expect_arch="hunyuan_video"),
    dict(id="hunyuan-vae", genre="VAE (HunyuanVideo 3D)",
         repo="Comfy-Org/HunyuanVideo_repackaged",
         file="split_files/vae/hunyuan_video_vae_bf16.safetensors",
         license="Tencent Hunyuan Community", gated=False,
         expect_kind="vae", expect_arch="vae_3d_16ch"),
    dict(id="wan22-ti2v", genre="Video DiT (Wan 2.2 TI2V-5B)",
         repo="Wan-AI/Wan2.2-TI2V-5B",
         file="diffusion_pytorch_model-00001-of-00003.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="unet_only", expect_arch="wan"),
    dict(id="chroma", genre="DiT (Chroma, FLUX.1 schnell derivative)",
         repo="lodestones/Chroma",
         file="chroma-unlocked-v16.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="unet_only", expect_arch="chroma"),
    dict(id="hidream", genre="DiT (HiDream-I1, MoE)",
         repo="HiDream-ai/HiDream-I1-Full",
         file="transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
         license="MIT", gated=False,
         expect_kind="unet_only", expect_arch="hidream"),
    dict(id="mochi", genre="Video DiT (Mochi 1)",
         repo="genmo/mochi-1-preview",
         file="dit.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="unet_only", expect_arch="mochi"),
    dict(id="zimage", genre="DiT (Z-Image Turbo)",
         repo="Tongyi-MAI/Z-Image-Turbo",
         file="transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="unet_only", expect_arch="z_image"),
    dict(id="wan21-t2v", genre="Video DiT (Wan 2.1)",
         repo="Wan-AI/Wan2.1-T2V-1.3B",
         file="diffusion_pytorch_model.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="unet_only", expect_arch="wan"),

    # --- Text Encoder ---
    dict(id="t5xxl", genre="Text Encoder (T5-XXL)",
         repo="comfyanonymous/flux_text_encoders",
         file="t5xxl_fp16.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="text_encoder", expect_arch="te_t5"),
    dict(id="t5xxl-fp8", genre="Text Encoder (T5-XXL, fp8 scaled)",
         repo="comfyanonymous/flux_text_encoders",
         file="t5xxl_fp8_e4m3fn_scaled.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="text_encoder", expect_arch="te_t5"),
    dict(id="clip-l", genre="Text Encoder (CLIP-L, standalone)",
         repo="comfyanonymous/flux_text_encoders",
         file="clip_l.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="text_encoder", expect_arch="te_clip"),

    # --- adapters and control models ---
    dict(id="lokr-sdxl", genre="LyCORIS LoKr (SDXL)",
         repo="LyliaEngine/USNR_STYLE_XL_lokr",
         file="USNR STYLE_XL_lokr.safetensors",
         license="CDLA-Permissive-2.0", gated=False,
         expect_kind="lora", expect_arch="sdxl"),
    dict(id="cn-sd15-ldm", genre="ControlNet (SD1.5, LDM / A1111 layout)",
         repo="monster-labs/control_v1p_sd15_qrcode_monster",
         file="control_v1p_sd15_qrcode_monster.safetensors",
         license="OpenRAIL++", gated=False,
         expect_kind="controlnet", expect_arch="sd15"),
    dict(id="cn-sd15-diffusers", genre="ControlNet (SD1.5, diffusers layout)",
         repo="monster-labs/control_v1p_sd15_qrcode_monster",
         file="diffusion_pytorch_model.safetensors",
         license="OpenRAIL++", gated=False,
         expect_kind="controlnet", expect_arch="sd15"),
    dict(id="cn-lllite", genre="ControlNet-LLLite (SDXL)",
         repo="kohya-ss/controlnet-lllite",
         file="controllllite_v01032064e_sdxl_canny.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="controlnet", expect_arch="sdxl"),

    dict(id="ti-sdxl", genre="Textual Inversion (SDXL)",
         repo="FoodDesert/Boring_Embeddings",
         file="boring_sdxl_v1.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="embedding", expect_arch="ti_sdxl"),
    dict(id="ti-sd15", genre="Textual Inversion (SD1.x)",
         repo="FoodDesert/Boring_Embeddings",
         file="boring_e621_v4.safetensors",
         license="Apache-2.0", gated=False,
         expect_kind="embedding", expect_arch="ti_sd15"),

    # --- gated: only reachable if you accepted the licence and set HF_TOKEN ---
    dict(id="flux-schnell", genre="DiT (FLUX.1 schnell, single-file)",
         repo="black-forest-labs/FLUX.1-schnell",
         file="flux1-schnell.safetensors",
         license="Apache-2.0", gated=True,
         expect_kind="unet_only", expect_arch="flux"),
    dict(id="flux-dev", genre="DiT (FLUX.1 dev, single-file)",
         repo="black-forest-labs/FLUX.1-dev",
         file="flux1-dev.safetensors",
         license="FLUX.1 [dev] Non-Commercial License", gated=True,
         expect_kind="unet_only", expect_arch="flux"),
    dict(id="flux-transformer", genre="DiT (FLUX.1 schnell, diffusers layout, sharded)",
         repo="black-forest-labs/FLUX.1-schnell",
         file="transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
         license="Apache-2.0", gated=True,
         expect_kind="unet_only", expect_arch="flux"),
    dict(id="flux-vae", genre="VAE (FLUX.1 autoencoder)",
         repo="black-forest-labs/FLUX.1-schnell",
         file="ae.safetensors",
         license="Apache-2.0", gated=True,
         expect_kind="vae", expect_arch="vae_sd_2d"),
    dict(id="sd35-medium", genre="MMDiT (SD 3.5 medium, DiT + VAE, no text encoder)",
         repo="stabilityai/stable-diffusion-3.5-medium",
         file="sd3.5_medium.safetensors",
         license="Stability AI Community License", gated=True,
         expect_kind="backbone_vae", expect_arch="sd3"),
]


def get_header(t):
    """キャッシュがあれば使い、なければ Range 取得する。"""
    cache = HEADER_DIR / f"{t['id']}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8")), True
    header, nbytes, _ = fetch_header(t["repo"], t["file"])
    HEADER_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(header, ensure_ascii=False), encoding="utf-8")
    print(f"    fetched {nbytes/1024:.0f} KB")
    return header, False


def analyze_header(header):
    """Run stinspect's detection logic directly on a header dict."""
    metadata = header.get("__metadata__") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    modules, prefix_components = stinspect.build_modules(header)
    dialect, _ = stinspect.detect_adapter(header)
    components = stinspect.detect_components(modules, prefix_components)
    ctx = stinspect.detect_context_dim(modules)
    kind = stinspect.classify_file_kind(dialect, components, modules)
    archs = stinspect.score_architectures(modules, ctx, metadata, kind)
    strong = [a for a in archs if a["score"] >= stinspect.MIN_ARCH_SCORE]
    return dict(kind=kind, dialect=dialect, components=components,
                ctx=ctx, archs=archs, strong=strong, modules=modules,
                metadata=metadata)


def main(argv):
    filt = argv[0].lower() if argv else None
    ok = ng = skip = 0
    for t in TARGETS:
        if filt and filt not in t["id"].lower():
            continue
        print("=" * 78)
        print(f"{t['id']}  [{t['genre']}]")
        print(f"  {t['repo']} :: {t['file']}   ({t['license']}"
              f"{', gated' if t['gated'] else ''})")
        try:
            header, cached = get_header(t)
        except Exception as e:
            print(f"    [could not fetch] {e}")
            skip += 1
            continue

        a = analyze_header(header)
        got_kind = a["kind"]
        got_arch = a["strong"][0]["arch"]["id"] if a["strong"] else None

        kind_ok = (got_kind == t["expect_kind"])
        arch_ok = (got_arch == t["expect_arch"])
        mark = "OK " if (kind_ok and arch_ok) else "NG "
        if kind_ok and arch_ok:
            ok += 1
        else:
            ng += 1

        print(f"  {mark} kind {got_kind}"
              + ("" if kind_ok else f"   <- expected {t['expect_kind']}"))
        print(f"      arch {got_arch}"
              + ("" if arch_ok else f"   <- expected {t['expect_arch']}"))
        if a["dialect"]:
            print(f"      dialect {a['dialect']['id']}")
        if a["ctx"]:
            print(f"      cross-attn width {dict(a['ctx'])}")
        print(f"      components {[c['id'] for c in a['components']]}")
        if a["strong"]:
            for ev in a["strong"][0]["evidence"][:3]:
                print(f"      evidence {stinspect.evidence_text(ev, 'en')}")
        elif a["archs"]:
            print("      weak candidates "
                  + str([(x["arch"]["id"], x["score"]) for x in a["archs"][:2]]))
        if a["metadata"]:
            print(f"      metadata {sorted(a['metadata'])[:6]}")
        print()

    print("=" * 78)
    print(f"OK {ok} / NG {ng} / could not fetch {skip}")
    if skip:
        print("Gated repositories return 401 unless you have accepted their licence "
              "on Hugging Face and set HF_TOKEN.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
