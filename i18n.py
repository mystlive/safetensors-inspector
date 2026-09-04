# -*- coding: utf-8 -*-
"""Message catalogue for stinspect.

Rule descriptions live in rules.py as {"en": ..., "ja": ...} dicts.
UI labels live here.
"""

LANGS = ("en", "ja")


def T(en, ja):
    """Bilingual string used throughout rules.py."""
    return {"en": en, "ja": ja}


def tr(value, lang):
    """Resolve a bilingual dict (or a plain string) for the given language."""
    if isinstance(value, dict):
        return value.get(lang) or value.get("en") or next(iter(value.values()), "")
    return value


LABELS = {
    "en": {
        "label_type": "Type",
        "label_base": "Base",
        "label_evidence": "Evidence",
        "label_caveat": "Caveat",
        "label_alt": "Also",
        "label_parts": "Contents",
        "label_targets": "Targets",
        "label_strength": "Strength",
        "label_dialect": "Dialect",
        "label_quant": "Quant",
        "label_metadata": "Metadata",
        "label_placement": "Place in",
        "label_keys": "Keys",
        "label_triggers": "Triggers",
        "hits": "{n} matches",

        "unreadable": "[unreadable]",
        "confidence": "confidence",
        "conf_high": "high",
        "conf_medium": "medium",
        "conf_low": "low",

        "verified_measured": "",
        "verified_derived": " [derived from measurement]",
        "verified_unverified": " [unverified / inferred]",

        "kind_checkpoint": "Full checkpoint (UNet/DiT + text encoder + VAE)",
        "kind_unet_only": "Diffusion backbone only (UNet / DiT). Text encoder and VAE needed separately",
        "kind_text_encoder": "Text encoder only",
        "kind_vae": "VAE only",
        "kind_controlnet": "ControlNet",
        "kind_embedding": "Textual Inversion / embedding",
        "kind_unknown": "Not identified",

        "base_unknown": "not identified",
        "base_weak": "weak match (not enough to conclude)",
        "base_hint": "Run with --keys to look at the key names, or add a rule to rules.py",

        "size_mismatch": ("[!] Size mismatch: expected {expected}, actual {actual}"
                          " - the download may be incomplete or the file corrupt"),
        "no_metadata": "none (the producing tool wrote nothing)",
        "thin_metadata": "only {n} low-value item(s): {keys}",
        "more_items": "({n} more: {keys}{ellipsis}  use --meta to show all)",
        "omitted": "({n} chars, omitted - use --meta for the full value)",
        "triggers": "Trigger words (most frequent training tags)",

        "rank_single": "rank {rank}",
        "rank_mixed": "mixed rank: {detail}",
        "alpha_single": "alpha {alpha}",
        "alpha_mixed": "mixed alpha: {detail}",
        "layers": "{n} layers",

        "naming_mix_1": "LDM naming (input_blocks) and diffusers naming (down_blocks) are both present",
        "naming_mix_2": "likely a file that ships both key sets for compatibility",

        "target_unet": "UNet / DiT",
        "target_te": "Text encoder",
        "target_vae": "VAE",

        "quant_fp8_scaled": "fp8 scaled (ComfyUI layout, carries scale_weight)",
        "quant_fp8": "fp8 (no scale correction)",
        "quant_int8": "possibly 8-bit quantized (contains U8 / I8 tensors)",

        "summary": "{n} file(s) analysed",
        "not_found": "no safetensors found",
        "skip": "[skip] not found: {path}",
        "wrote": "Wrote results to: {path}",
        "wrote_csv": "Wrote CSV to: {path}",

        "err_short": "file is under 8 bytes; not a safetensors",
        "err_zero": "header length is 0",
        "err_huge": "implausible header length ({n} bytes); not a safetensors, or corrupt",
        "err_truncated": "header is cut short (download may be incomplete)",
        "err_json": "header is not valid JSON: {err}",
        "err_notdict": "header is not an object",

        "csv_name": "file", "csv_kind": "type", "csv_base": "base",
        "csv_conf": "confidence", "csv_rank": "rank", "csv_size": "size",
        "csv_tensors": "tensors", "csv_params": "params", "csv_path": "path",
        "csv_error": "error",

        "help_desc": "Identify what a safetensors file is: full model, LoRA, VAE or text encoder, and which base it needs",
        "help_targets": "files or folders (default: current folder)",
        "help_recursive": "walk folders recursively",
        "help_meta": "print all metadata instead of the highlights",
        "help_keys": "also print sample key names",
        "help_json": "emit JSON",
        "help_csv": "write a summary CSV (UTF-8 with BOM)",
        "help_out": "write results to a file (UTF-8 with BOM, so it opens cleanly in Notepad and Excel)",
        "help_lang": "output language (default: en)",
    },
    "ja": {
        "label_type": "種別",
        "label_base": "ベース",
        "label_evidence": "根拠",
        "label_caveat": "注意",
        "label_alt": "他の候補",
        "label_parts": "構成",
        "label_targets": "適用先",
        "label_strength": "強度",
        "label_dialect": "キー方言",
        "label_quant": "量子化",
        "label_metadata": "メタデータ",
        "label_placement": "配置",
        "label_keys": "キー例",
        "label_triggers": "トリガー",
        "hits": "{n} 件",

        "unreadable": "[読めない]",
        "confidence": "確度",
        "conf_high": "高",
        "conf_medium": "中",
        "conf_low": "低",

        "verified_measured": "",
        "verified_derived": " [実測から導出]",
        "verified_unverified": " [未検証・推定]",

        "kind_checkpoint": "モデル本体（UNet/DiT + Text Encoder + VAE を含む完全なチェックポイント）",
        "kind_unet_only": "モデル本体の一部（UNet / DiT のみ。Text Encoder と VAE は別途必要）",
        "kind_text_encoder": "Text Encoder 単体",
        "kind_vae": "VAE 単体",
        "kind_controlnet": "ControlNet",
        "kind_embedding": "Textual Inversion / Embedding",
        "kind_unknown": "判別不能",

        "base_unknown": "判別できず",
        "base_weak": "弱い一致（断定には足りない）",
        "base_hint": "--keys でキー名を見て手がかりを探すか、rules.py にルールを追加すること",

        "size_mismatch": ("[!] ファイルサイズ不一致: 期待 {expected} / 実際 {actual}"
                          " — ダウンロード未完了または破損の疑い"),
        "no_metadata": "なし（作成ソフトが書き込んでいない）",
        "thin_metadata": "情報の乏しい {n} 項目のみ: {keys}",
        "more_items": "(他 {n} 項目: {keys}{ellipsis}  --meta で全表示)",
        "omitted": "({n} 文字・省略。--meta で全文表示)",
        "triggers": "トリガー候補 (学習時のタグ頻度上位)",

        "rank_single": "rank {rank}",
        "rank_mixed": "rank 混在: {detail}",
        "alpha_single": "alpha {alpha}",
        "alpha_mixed": "alpha 混在: {detail}",
        "layers": "{n}層",

        "naming_mix_1": "LDM 命名 (input_blocks 系) と diffusers 命名 (down_blocks 系) が混在",
        "naming_mix_2": "互換性のため両方のキーを収録したファイルとみられる",

        "target_unet": "UNet / DiT",
        "target_te": "Text Encoder",
        "target_vae": "VAE",

        "quant_fp8_scaled": "fp8 scaled (ComfyUI 形式。scale_weight を伴う)",
        "quant_fp8": "fp8 (スケール補正なし)",
        "quant_int8": "8bit 量子化の可能性（U8 / I8 テンソルを含む）",

        "summary": "{n} ファイルを解析",
        "not_found": "safetensors が見つからない",
        "skip": "[skip] 見つからない: {path}",
        "wrote": "結果を書き出した: {path}",
        "wrote_csv": "CSV を書き出した: {path}",

        "err_short": "ファイルが 8 バイト未満。safetensors ではない",
        "err_zero": "ヘッダ長が 0",
        "err_huge": "ヘッダ長が異常 ({n} bytes)。safetensors でないか、破損している",
        "err_truncated": "ヘッダが途中で切れている（ダウンロード未完了の可能性）",
        "err_json": "ヘッダが JSON として読めない: {err}",
        "err_notdict": "ヘッダが辞書でない",

        "csv_name": "ファイル名", "csv_kind": "種別", "csv_base": "ベース推定",
        "csv_conf": "確度", "csv_rank": "rank", "csv_size": "サイズ",
        "csv_tensors": "tensors", "csv_params": "params", "csv_path": "パス",
        "csv_error": "エラー",

        "help_desc": "safetensors の素性（本体か LoRA か、必要なベースは何か）を判別する",
        "help_targets": "ファイルまたはフォルダ（省略時はカレントフォルダ）",
        "help_recursive": "フォルダを再帰的に走査",
        "help_meta": "メタデータを省略せず全部出す",
        "help_keys": "キー名のサンプルも出す",
        "help_json": "JSON で出力",
        "help_csv": "一覧を CSV に書き出す（UTF-8 BOM 付き）",
        "help_out": "結果をファイルに書き出す（UTF-8 BOM 付き。メモ帳や Excel でそのまま開ける）",
        "help_lang": "出力言語（既定: en）",
    },
}


def L(lang, key, **kw):
    """Look up a UI label, formatting it with kw when needed."""
    s = LABELS.get(lang, LABELS["en"]).get(key)
    if s is None:
        s = LABELS["en"].get(key, key)
    return s.format(**kw) if kw else s
