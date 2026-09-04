# safetensors-inspector

素性の分からなくなった `.safetensors` が何なのか — モデル本体か、LoRA か、VAE か、Text Encoder か — そして必要なベースモデルは何かを、重みを読み込まずに判別する。

ダウンロードしたまま、どれがどれだか分からなくなったフォルダのための道具。

English: [README.md](README.md)

```
==============================================================================
sd_xl_offset_example-lora_1.0.safetensors
  models/loras/sd_xl_offset_example-lora_1.0.safetensors
  47.3 MB / 2364 tensors / 24.6M params / F16:2364
------------------------------------------------------------------------------
  種別        LoRA (kohya / sd-scripts 形式)
              A1111 / ComfyUI / Forge がそのまま読める最も一般的な LoRA 形式
  ベース      SDXL 系 (SDXL 1.0 / Illustrious / Pony / NoobAI / Animagine ほか)   確度 高
              根拠: SDXL の深いブロックに transformer が乗る構成 (100 件)
              根拠: cross-attention の入力次元 = 2048
              根拠: メタデータの宣言と一致: stable-diffusion-xl-v1-base/lora
              注意: Illustrious / Pony / NoobAI / Animagine などの派生は重みの構造が SDXL 1.0 と完全に同一のため、
                    構造だけでは区別できない。区別にはメタデータかファイル名が要る
  構成        UNet (LDM / SAI 命名)
  適用先      UNet / DiT
  強度        rank 8   alpha 1
  メタデータ  学習時のベース: sdxl_base_v0-9
              rank (network_dim): 8
              alpha: 1
              学習画像数: 7412
              宣言アーキテクチャ: stable-diffusion-xl-v1-base/lora
              タイトル: SDXL 1.0 Official Offset Example LoRA
              (他 49 項目: ... --meta で全表示)
  トリガー    contrast (7412)
  配置        ComfyUI: models/loras
              LoraLoader ノード。ベースモデルを合わせること
```

*(README 用にメタデータ部分を削っている。実際にはもっと出る)*

## 速い理由

safetensors はファイル先頭に JSON ヘッダを持つ。

```
[8 バイト: ヘッダ長 N (unsigned little-endian 64bit)]
[N バイト: UTF-8 JSON]
[テンソル本体]
```

このツールが読むのはそのヘッダだけ（あとは alpha スカラーの数バイト）。
6GB のチェックポイントでも数百 KB の読み取りで判別できるので、モデルフォルダ全体の走査が数秒で終わる。

## 導入

依存ライブラリなし。Python 3.8 以降の標準ライブラリのみ。PyTorch も `safetensors` パッケージも不要。

```bash
git clone https://github.com/mystlive/safetensors-inspector
cd safetensors-inspector
python stinspect.py --help
```

## 使い方

```bash
# 1 ファイル
python stinspect.py path/to/mystery.safetensors --lang ja

# モデルフォルダごと
python stinspect.py path/to/ComfyUI/models -r --lang ja

# レポートをファイルに（UTF-8 BOM 付き。メモ帳でも化けない）
python stinspect.py path/to/models -r --lang ja -o report.txt

# 所持モデルの一覧を CSV に
python stinspect.py path/to/models -r --lang ja --csv inventory.csv
```

| オプション | 動作 |
| --- | --- |
| `-r`, `--recursive` | フォルダを再帰的に走査 |
| `-o PATH` | レポートをファイルに書き出す（UTF-8 BOM 付き） |
| `--csv PATH` | 一覧表を CSV に（UTF-8 BOM 付き。Excel でそのまま開ける） |
| `--meta` | メタデータを省略せず全部出す |
| `--keys` | キー名のサンプルも出す（判別できなかったファイルの調査用） |
| `--json` | JSON で出力 |
| `--lang {en,ja}` | 出力言語（既定は `en`） |

ファイルに残すときは `>` ではなく `-o` を使う。`>` は BOM なし UTF-8 で書くため、メモ帳などが Shift_JIS と誤認して化ける。

## 判別できること

- **種別** — 完全なチェックポイント / 拡散モデル本体のみ / LoRA / VAE / Text Encoder / ControlNet / Embedding
- **ベースモデル** — SD1.x、SDXL、Qwen-Image、Wan、Anima など。判定の根拠付き
- **LoRA の詳細** — rank、alpha、Text Encoder に効くかどうか、キー形式（形式によっては A1111 / ComfyUI がそのままでは読めない）
- **配置先** — ComfyUI のフォルダと、読み込むノード
- **トリガーワード** — 学習時のタグ頻度が記録されていれば
- **破損** — ダウンロード未完了やヘッダ破損を、サイズの照合で検出

判定にはすべて根拠が付く。実ファイルで未検証のルールは `[未検証・推定]` と表示されるので、どこまで信じてよいか分かる。

## 判別できないこと

構造が同一のものはヘッダから区別できない。ツールは推測せず、そう言う。

| 区別できない組 | 理由 |
| --- | --- |
| SDXL 1.0 / Illustrious / Pony / NoobAI / Animagine | UNet も Text Encoder も形が完全に同一。違うのは値だけ |
| Qwen-Image / Qwen-Image-Edit | DiT 構造が同一 |
| Wan 2.x の VAE / Qwen-Image の VAE | 同系の 3D VAE |
| SD1.x 用 VAE / SDXL 用 VAE | 構造が同一 |
| FLUX.1 dev / schnell | `guidance_in` の有無が唯一の手がかり |

メタデータ（`ss_base_model_version`、`modelspec.architecture`）が残っていればそれを使い、使ったことも表示する。残っていない場合は、ファイル名か配布元を当たるしかない。

## ルールの仕組み

`rules.py` に判定ルールをデータとして置き、`stinspect.py` にロジックを置いてある。新しいモデルへの対応でロジックを触る必要はない。

キーは照合前に正規化される。接頭辞（`lora_unet_`、`model.diffusion_model.`、`net.` など）とテンソルのサフィックス（`.weight`、`.lora_down.weight` など）を剥がし、区切り文字を揃える。

```
kohya      lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight
diffusers  unet.down_blocks.0.resnets.0.conv1.lora_down.weight
どちらも →  down_blocks_0_resnets_0_conv1
```

これで 1 つのルールが全方言をカバーする。アーキテクチャの追加は 1 エントリで済む。

```python
{
    "id": "my_model",
    "name": T("My Model", "My Model"),
    "verified": "measured",
    "signals": [
        (r"^some_distinctive_key_\d+", 4, T("why this is evidence", "根拠の説明")),
    ],
    "context_dims": [],     # cross-attention 次元が決まっているなら書く
    "veto": [],             # これがあれば候補から外す
    "note": T("", ""),      # 区別できない点があれば書く
    "comfy_dir": "diffusion_models",
},
```

ルールを書くためにキー名を調べるには:

```bash
python tools/probe_header.py path/to/file.safetensors
```

## 検証

ルールは記憶からでなく実ファイルで検証している。`tools/verify_rules.py` は公開モデル 22 件のヘッダだけを HTTP Range で取得し（重みはダウンロードせず、再配布もしない）、期待する判定結果と照合する。

```bash
python tools/verify_rules.py
```

使用したモデル、そこから確定した指紋、未検証のまま残っている項目は [docs/key-reference.md](docs/key-reference.md) にまとめてある。

FLUX や SD3.5 は gated リポジトリにあるため、ルールは `unverified` のまま。検証するには Hugging Face で自分でライセンスに同意し、`HF_TOKEN` を設定する。設定すれば `tools/verify_rules.py` が対象に含める。

## 出力の読み方

[docs/guide.ja.md](docs/guide.ja.md) に、各行の意味、rank と alpha から実際の倍率が決まる仕組み、判別できなかったときの調べ方、LoRA が効かないときの切り分け方をまとめてある。

## ライセンス

MIT。[LICENSE](LICENSE) を参照。
