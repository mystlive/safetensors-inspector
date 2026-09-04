# 出力の読み方と、そこからの使い方

`stinspect.py` の出力を、上から順に何を見て、何をすればよいか。

## 目次

1. [読む順序](#読む順序)
2. [各行の意味](#各行の意味)
3. [種別ごとの対処](#種別ごとの対処)
4. [rank と alpha の読み方](#rank-と-alpha-の読み方)
5. [判別できなかったときの調べ方](#判別できなかったときの調べ方)
6. [落とし穴](#落とし穴)

---

## 読む順序

出力は多いが、実際に見るのは 3 行だけで足りることが多い。

```
  種別      ← ①これは何のファイルか
  ベース    ← ②何と組み合わせて使うのか
  配置      ← ③どこに置くか
```

この 3 つで「使える状態」になる。
LoRA の場合はこれに **強度** 行が加わる。

残りの行（構成、キー方言、量子化、メタデータ）は、
うまく動かないときの原因究明に使う。

---

## 各行の意味

### 種別

そのファイルが何であるか。取り得る値は 6 種類。

| 表示 | 意味 | 単体で動くか |
| --- | --- | --- |
| モデル本体（... 完全なチェックポイント） | UNet/DiT + Text Encoder + VAE が全部入り | **動く** |
| モデル本体の一部（UNet / DiT のみ） | 拡散モデル部分だけ | 動かない。TE と VAE を別途用意 |
| LoRA (...形式) | 差分。ベースに重ねて使う | 動かない。ベースが要る |
| VAE 単体 | 潜在→画像の変換器だけ | 動かない。本体に添える |
| Text Encoder 単体 | プロンプト解釈器だけ | 動かない。本体に添える |
| Textual Inversion / Embedding | 単語の埋め込みだけ | 動かない。プロンプトで呼ぶ |

LoRA には形式名が続く。この違いは実用上重要（後述の「キー方言」）。

### ベース ／ 確度

そのファイルを動かすために必要なベースモデル。

| 確度 | 意味 | 扱い |
| --- | --- | --- |
| 高 | cross-attention の次元一致など、構造上ほぼ確定 | 信じてよい |
| 中 | 特徴的なキー名が複数一致した | ほぼ正しいが、動かなければ疑う |
| 低 | 弱い一致 | 参考程度 |
| 判別できず | 該当ルールなし | 自分で調べる（後述） |

名前の後ろに印が付くことがある。

| 印 | 意味 |
| --- | --- |
| （印なし） | 手元の実ファイルで確認済みのルールによる判定 |
| `[実測から導出]` | 確認済みの事実から論理的に導いた判定。ほぼ確実だが実物では未確認 |
| `[未検証・推定]` | 実物がなく、公開実装のキー命名からの推定。**外れることがある** |

`根拠:` の行はなぜそう判定したかの説明。
`注意:` の行は「ここまでは分かるが、この先は構造からは分からない」という限界。

たとえば SDXL 系はこう出る。

```
  ベース    SDXL 系 (SDXL 1.0 / Illustrious / Pony / NoobAI / Animagine ほか)   確度 高
            根拠: cross-attention の入力次元 = 2048
            注意: Illustrious / Pony / NoobAI / Animagine などの派生は重みの構造が
                  SDXL 1.0 と完全に同一のため、構造だけでは区別できない
```

**「SDXL 系」までは確実、その中のどれかは不明**、という読み方をする。
実用上は、手持ちの SDXL 系チェックポイントを順に試すのが早い。
アニメ系の絵柄で学習された LoRA なら Illustrious 系、という当たりの付け方はできるが、
それはファイル名や配布元から推測しているのであって、ツールが判定した事実ではない。

### 構成

ファイルの中身の内訳。「種別」の判断根拠にあたる。

`UNet (LDM / SAI 命名)` と `UNet (diffusers 命名)` の違いはキー名の流儀であって、
中身の性能差ではない。

### 適用先

LoRA がどこに効くか。

- `UNet / DiT` … 絵柄・構図・被写体に効く。大半の LoRA はこれ
- `Text Encoder` … 特定の語の解釈を変える。トリガーワードを持つ LoRA に多い
- `UNet / DiT + Text Encoder` … 両方

Text Encoder に効く LoRA は、ComfyUI の LoraLoader で
CLIP 側も接続しないと本来の効果が出ない
（`LoraLoaderModelOnly` を使うと Text Encoder 側が無視される）。

### 強度

LoRA の rank と alpha。詳細は[後述](#rank-と-alpha-の読み方)。

### キー方言

`LDM 命名と diffusers 命名が混在` と出た場合、
1 つのファイルに 2 系統のキー名が入っている。
互換性のために両方収録したか、複数の LoRA をマージしたもの。
読み込み自体は問題ないことが多い。

`LoRA (ドット区切り diffusers 形式 / OneTrainer 内部保存)` と出た場合は要注意。
キー名が `lora_unet_` 形式でないため、A1111 や ComfyUI が読めないことがある。
OneTrainer なら `workspace/.../backup/` ではなく、
「保存」で書き出した方のファイルを使う。

### 量子化

`fp8 scaled (ComfyUI 形式)` … 重みを fp8 に落とし、層ごとの補正係数を併せ持つ形式。
VRAM は減るが、対応した読み込み側が要る。

### メタデータ

学習時に書き込まれた情報。**構造判定より信頼できる場合がある**。
特に次の 2 つはベース特定の決め手になる。

- `学習時のベース` (`ss_base_model_version`) … 学習に使ったベースそのもの
- `宣言アーキテクチャ` (`modelspec.architecture`) … 作成ソフトが宣言した種別

`マージ元` が出ていれば、何を混ぜたモデルか分かる。

`トリガー` 行が出た場合、それは学習時のタグ頻度上位。
LoRA を効かせるためにプロンプトへ入れる語の候補になる。
ただし頻度が高い＝トリガーとは限らない（`1girl` のような汎用タグも上位に来る）。

### 配置

ComfyUI のどのフォルダに置くか。
ComfyUI は `models/diffusion_models`（旧名 `unet`）と
`models/text_encoders`（旧名 `clip`）を内部で読み替えるため、
旧名のフォルダに置いても読まれる。

---

## 種別ごとの対処

### モデル本体（完全なチェックポイント）

`models/checkpoints` に置く。Load Checkpoint ノードで読む。それだけで動く。

A1111 / Forge なら `models/Stable-diffusion`。

### モデル本体の一部（UNet / DiT のみ）

`models/diffusion_models` に置く。**これ単体では動かない**。
「ベース」の判定を見て、対応する Text Encoder と VAE を揃える。

| ベース判定 | 追加で要るもの |
| --- | --- |
| Qwen-Image / Qwen-Image-Edit | Qwen2.5-VL 7B の Text Encoder ＋ Qwen-Image VAE |
| Anima | Qwen3-0.6B 系の Text Encoder ＋ 対応 VAE |
| FLUX.1 | T5-XXL ＋ CLIP-L ＋ Flux VAE (ae.safetensors) |

手持ちに何があるかは、同じツールで `models` フォルダごと走査すれば分かる。

```bash
python stinspect.py "path/to/ComfyUI/models" -r --lang ja -o inventory.txt
```

`種別: Text Encoder 単体` や `VAE 単体` のファイルが一覧に出るので、
ベース判定が一致するものを組み合わせる。

### LoRA

`models/loras` に置く。LoraLoader で読む。

ベースが合っていないと、効かないか、絵が破綻する。
「ベース」行が `SDXL 系` なら SDXL 系チェックポイントと、
`Qwen-Image` なら Qwen-Image 本体と組み合わせる。

読み込み時にエラーが出る場合は「キー方言」行を確認する。

### VAE 単体 / Text Encoder 単体

`models/vae` / `models/text_encoders` に置き、本体と組み合わせる。

VAE を差し替えると、色味と細部の再現が変わる。
本体に VAE が内蔵されている（種別が「完全なチェックポイント」）場合、
外部 VAE を指定しなければ内蔵のものが使われる。

### 判別できず

[後述](#判別できなかったときの調べ方)。

---

## rank と alpha の読み方

```
  強度      rank 16   alpha 16
```

### 何を意味するか

- **rank**（`network_dim`）… LoRA の表現力。大きいほど多くを学習できるがファイルも大きい
- **alpha** … 学習時に決めた減衰係数

実際に適用される倍率は `alpha ÷ rank` になる。
これは学習側（kohya sd-scripts）と推論側（ComfyUI）の両方で同じ式が使われている。

学習側 `networks/lora.py`:

```python
alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
self.scale = alpha / self.lora_dim
```

推論側 ComfyUI `comfy/weight_adapter/lora.py`:

```python
if v[2] is not None:
    alpha = v[2] / mat2.shape[0]
else:
    alpha = 1.0
...
weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
```

`strength` が UI で設定する強度。最終的な倍率は **UI 強度 × (alpha ÷ rank)** になる。

### 読み方の例

| 表示 | alpha ÷ rank | 意味 |
| --- | --- | --- |
| rank 16 / alpha 16 | 1.0 | 標準。UI 強度 1.0 がそのまま効く |
| rank 64 / alpha 8 | 0.125 | 控えめに設計されている |
| rank 4 / alpha 1 | 0.25 | 控えめ |
| rank 32 / alpha 32 | 1.0 | 標準 |

`alpha` が表示されない（PEFT / ai-toolkit 形式など）場合、
ComfyUI は倍率 1.0 として扱う（上のコードの `else: alpha = 1.0`）。

### 注意

これは**学習時にどう設定されたか**であって、**推奨強度ではない**。
`alpha ÷ rank` が小さいからといって UI 強度を上げるべきとは限らない。
配布元に推奨値があればそちらに従う。

`rank 混在: 8(743層), 4(346層)` のように出る場合、
層ごとに rank が違う（ブロック別に dim を変えて学習した LoRA）。
異常ではない。

---

## 判別できなかったときの調べ方

```
  ベース    判別できず
            弱い一致（断定には足りない）: Wan 2.x (動画生成) (score 2: ...)
```

### 手順 1: メタデータを全部見る

```bash
python stinspect.py "謎のファイル.safetensors" --lang ja --meta -o meta.txt
```

省略されていた項目が全部出る。
`ss_sd_model_name`、`ss_network_args`、`workflow`（ComfyUI の作業記録）などに
手がかりが残っていることがある。

### 手順 2: キー名を見る

```bash
python stinspect.py "謎のファイル.safetensors" --lang ja --keys
```

キー名の先頭部分（`double_blocks`、`joint_blocks`、`blocks` など）が
アーキテクチャの指紋になる。この文字列でモデル名を検索すると当たることが多い。

さらに詳しく見るなら調査用スクリプトを使う。

```bash
python tools/probe_header.py "謎のファイル.safetensors"
```

キーの階層別集計、shape、dtype 分布が出る。

### 手順 3: ルールを追加する

判明したら `rules.py` の `ARCHITECTURES` に 1 エントリ足す。
書き方は [../README.ja.md](../README.ja.md) の「拡張のしかた」にある。
次から自動で判定されるようになる。

### 判別できないのが正常な場合

画像生成用でないモデル（DINOv3、MoGe などの視覚モデル、3D 生成モデル）は
`判別できず` または `unknown` になる。このツールは画像生成系に限定してある。

---

## 落とし穴

### ファイルサイズ不一致が出た

```
  [!] ファイルサイズ不一致: 期待 6937890000 / 実際 4200000000
      — ダウンロード未完了または破損の疑い
```

ヘッダは読めているが、テンソル本体が足りない。ダウンロードし直す。

### 「SDXL 系」と出たのに絵が崩れる

派生モデル（Illustrious / Pony / NoobAI）の違いが原因のことが多い。
構造は同じでも学習内容が違うため、Pony 用 LoRA を Illustrious に当てると
効きが悪かったり破綻したりする。ツールでは区別できない。

メタデータの `学習時のベース` か `マージ元` を見る。
それも無ければ、配布元のページを当たる。

### LoRA が全く効かない

順に疑う。

1. ベースが違う（「ベース」行と、使っているチェックポイントを照合）
2. キー方言が対応外（「種別」行が `ドット区切り diffusers 形式` になっていないか）
3. Text Encoder に効く LoRA なのに CLIP 側を繋いでいない（「適用先」行を確認）
4. トリガーワードを入れていない（「トリガー」行を確認）

### 同じ内容のファイルが複数ある

OneTrainer や kohya は学習途中のスナップショットを大量に吐く。
CSV で一覧にすると重複が見つけやすい。

```bash
python stinspect.py "path/to/models" -r --lang ja --csv all.csv
```

`tensors` 数と `params` が同じものは、同じ学習の別ステップである可能性が高い。
