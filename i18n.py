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
        "verified_derived": " [derived, not directly measured]",
        "verified_unverified": " [unverified / inferred]",

        "kind_checkpoint": "Full checkpoint (UNet/DiT + text encoder + VAE)",
        "kind_unet_only": "Diffusion backbone only (UNet / DiT). Text encoder and VAE needed separately",
        "kind_backbone_vae": "Diffusion backbone + VAE, no text encoder (load the text encoders separately)",
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
        "summary_title": "Summary - {n} file(s)",
        "summary_by_type": "By type",
        "summary_by_base": "By base model",
        "summary_damaged": "Unreadable - {n} file(s)",
        "summary_unresolved": "Not identified - {n} file(s), listed for next time",
        "unresolved_keys": "top-level keys",
        "unresolved_hint_1": "These need a rule. Run tools/probe_header.py on one to see its full key structure.",
        "unresolved_hint_2": "Pass --unresolved PATH to write the details to a file you can come back to.",
        "unresolved_hint_3": "Models outside image and video generation (depth estimators, 3D, vision backbones) land here too - those are out of scope by design.",
        # No leading "#": write_unresolved adds one for the file it writes, and
        # the HTML panel uses the bare title as its heading.
        "unresolved_file_title": "Unidentified files - {n}",
        "unresolved_file_intro": ("Everything needed to write a rule for these. Add one entry per\n"
                                  "architecture to ARCHITECTURES in rules.py, tagged \"unverified\"\n"
                                  "until you have checked it against the file itself."),
        "wrote_unresolved": "Wrote {n} unidentified file(s) to: {path}",
        "not_found": "no safetensors found",
        "skip": "[skip] not found: {path}",
        "wrote": "Wrote results to: {path}",
        "wrote_csv": "Wrote CSV to: {path}",
        "wrote_html": "Wrote HTML report to: {path}",

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

        # HTML report. "showing" keeps its placeholders: the browser fills them
        # in as the list is filtered.
        "html_title": "safetensors inspector",
        "html_search": "search file name, path or base",
        "html_all_kinds": "all types",
        "html_showing": "{n} of {total}",
        "html_show_more": "show more ({n} left)",
        "html_no_match": "nothing matches",

        # stgui.py, the launcher window.
        "gui_title": "safetensors inspector",
        "gui_target": "Folder",
        "gui_output": "Report",
        "gui_output_default": "leave empty to write to a temporary file",
        "gui_browse": "Browse...",
        "gui_recursive": "include subfolders",
        "gui_meta": "all metadata",
        "gui_keys": "sample keys",
        "gui_lang": "language",
        "gui_scan": "Scan",
        "gui_cancel": "Stop",
        "gui_pick_target": "Choose the folder to scan",
        "gui_pick_output": "Save the report as",
        "gui_need_target": "choose a folder first",
        "gui_collecting": "looking for safetensors...",
        "gui_progress": "{done} / {total}   {name}",
        "gui_none": "no safetensors found",
        "gui_done": "{n} file(s). Opened {path}",
        "gui_cancelled": "stopped",
        "gui_failed": "failed: {err}",

        "help_desc": "Identify what a safetensors file is: full model, LoRA, VAE or text encoder, and which base it needs",
        "help_targets": "files or folders (default: current folder)",
        "help_recursive": "walk folders recursively",
        "help_meta": "print all metadata instead of the highlights",
        "help_keys": "also print sample key names",
        "help_json": "emit JSON",
        "help_csv": "write a summary CSV (UTF-8 with BOM)",
        "help_html": "write a self-contained HTML report: a sortable, searchable table whose rows open into the full report",
        "help_out": "write results to a file (UTF-8 with BOM, so it opens cleanly in Notepad and Excel)",
        "help_unresolved": "write the files it could not identify to a file, with what is needed to add rules for them",
        "help_no_summary": "skip the summary at the end of a multi-file run",
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
        "verified_derived": " [導出・実測ではない]",
        "verified_unverified": " [未検証・推定]",

        "kind_checkpoint": "モデル本体（UNet/DiT + Text Encoder + VAE を含む完全なチェックポイント）",
        "kind_unet_only": "モデル本体の一部（UNet / DiT のみ。Text Encoder と VAE は別途必要）",
        "kind_backbone_vae": "モデル本体 + VAE（Text Encoder なし。別途読み込みが必要）",
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
        "summary_title": "サマリ — {n} ファイル",
        "summary_by_type": "種別ごと",
        "summary_by_base": "ベースモデルごと",
        "summary_damaged": "読めなかったもの — {n} ファイル",
        "summary_unresolved": "判別できず — {n} ファイル（次回に回す分）",
        "unresolved_keys": "トップレベルのキー",
        "unresolved_hint_1": "これらはルールの追加が要る。tools/probe_header.py にかけると全キー構造が見られる。",
        "unresolved_hint_2": "--unresolved PATH を付けると、後から見返せるようファイルに書き出す。",
        "unresolved_hint_3": "画像・動画生成用でないモデル（深度推定、3D 生成、視覚バックボーンなど）もここに出る。これらは対象外として想定どおり。",
        # 先頭の "#" は付けない。ファイルに書くときは write_unresolved が足し、
        # HTML のパネルは見出しとしてそのまま使う
        "unresolved_file_title": "判別できなかったファイル — {n} 件",
        "unresolved_file_intro": ("これらのルールを書くのに必要な情報。rules.py の ARCHITECTURES に\n"
                                  "アーキテクチャごとに 1 エントリ足す。実ファイルで確認するまでは\n"
                                  "\"unverified\" のままにしておくこと。"),
        "wrote_unresolved": "判別できなかった {n} 件を書き出した: {path}",
        "not_found": "safetensors が見つからない",
        "skip": "[skip] 見つからない: {path}",
        "wrote": "結果を書き出した: {path}",
        "wrote_csv": "CSV を書き出した: {path}",
        "wrote_html": "HTML レポートを書き出した: {path}",

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

        # HTML レポート。showing はプレースホルダのまま渡す（絞り込みに応じて
        # ブラウザ側が埋める）
        "html_title": "safetensors inspector",
        "html_search": "ファイル名・パス・ベースで検索",
        "html_all_kinds": "種別すべて",
        "html_showing": "{n} / {total} 件",
        "html_show_more": "続きを表示（残り {n} 件）",
        "html_no_match": "該当なし",

        # stgui.py（ランチャー画面）
        "gui_title": "safetensors inspector",
        "gui_target": "対象フォルダ",
        "gui_output": "出力先",
        "gui_output_default": "空欄なら一時ファイルに書く",
        "gui_browse": "参照...",
        "gui_recursive": "サブフォルダも見る",
        "gui_meta": "メタデータを全部",
        "gui_keys": "キー例も出す",
        "gui_lang": "言語",
        "gui_scan": "走査",
        "gui_cancel": "中止",
        "gui_pick_target": "走査するフォルダを選ぶ",
        "gui_pick_output": "レポートの保存先",
        "gui_need_target": "先にフォルダを選ぶ",
        "gui_collecting": "safetensors を探している...",
        "gui_progress": "{done} / {total}   {name}",
        "gui_none": "safetensors が見つからない",
        "gui_done": "{n} 件。{path} を開いた",
        "gui_cancelled": "中止した",
        "gui_failed": "失敗: {err}",

        "help_desc": "safetensors の素性（本体か LoRA か、必要なベースは何か）を判別する",
        "help_targets": "ファイルまたはフォルダ（省略時はカレントフォルダ）",
        "help_recursive": "フォルダを再帰的に走査",
        "help_meta": "メタデータを省略せず全部出す",
        "help_keys": "キー名のサンプルも出す",
        "help_json": "JSON で出力",
        "help_csv": "一覧を CSV に書き出す（UTF-8 BOM 付き）",
        "help_html": "自己完結の HTML レポートを書き出す（並べ替え・検索できる一覧。行を開くと詳細）",
        "help_out": "結果をファイルに書き出す（UTF-8 BOM 付き。メモ帳や Excel でそのまま開ける）",
        "help_unresolved": "判別できなかったファイルを、ルール追加に必要な情報つきで書き出す",
        "help_no_summary": "複数ファイル走査時の末尾サマリを出さない",
        "help_lang": "出力言語（既定: en）",
    },
}


def L(lang, key, **kw):
    """Look up a UI label, formatting it with kw when needed."""
    s = LABELS.get(lang, LABELS["en"]).get(key)
    if s is None:
        s = LABELS["en"].get(key, key)
    return s.format(**kw) if kw else s
