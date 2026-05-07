# 学習・評価レポート

日付: 2026-05-03

## 概要

依頼された「学習から精度評価まで」の一連の実行は、この実行環境では CUDA が利用できないため完了できませんでした。

- `nvidia-smi` は `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.` で失敗しました。
- `torch.cuda.is_available()` は `False` でした。
- `torch.cuda.device_count()` は `0` でした。

一方で、次回の GPU 環境での学習に向けて、コードと設定は以下の状態に整えています。

- SFT は既定で completion-only loss を使います。
- Qwen3.5 では chat template を既定で使います。
- Qwen 系設定は `Qwen/Qwen3.5-9B` に統一しています。
- `detector` + `corrector` は論文準拠の 2 段階再現経路として維持し、`Edit-Only` は改善比較用として扱います。

## データセット規模

fold 0 の前処理済み件数は以下です。

| task | train 件数 | test 件数 |
|---|---:|---:|
| detector | 5538 | 1385 |
| corrector | 5538 | 1385 |
| one_stage | 5538 | 1385 |
| edit_only | 5538 | 1385 |

## 実行できた評価

### Raw OCR ベースライン

fold 0 の `one_stage` test 行について、`raw_ocr_text` をそのまま `restored_text` として使うベースライン予測を生成しました。

成果物:

- `results/baselines/fold_0_raw_ocr_predictions.jsonl`
- `results/baselines/fold_0_raw_ocr_metrics_cli.json`

評価値:

| 指標 | 値 |
|---|---:|
| BLEU | 14.914826 |
| CRR | 0.561640 |
| WRR | 0.532083 |

### 既存 detector smoke 予測

既存の 1 件だけの detector smoke 予測を評価しました。

成果物:

- `results/detector_gpt_oss_20b/fold_0_predictions.smoke.jsonl`
- `results/detector_gpt_oss_20b/fold_0_predictions.smoke.char_metrics.json`

評価値:

| 指標 | 値 |
|---|---:|
| accuracy | 0.938650 |
| precision | 0.000000 |
| recall | 0.000000 |
| F1 | 0.000000 |

解釈: accuracy は一見高く見えますが、この smoke 出力は有効な誤り span を検出できていません。precision / recall / F1 がすべて 0 であるため、この成果物の detector は機能していないと判断すべきです。

## 評価処理の修正

`src/koten_refiner/metrics.py` は、以前は `evaluate.load("sacrebleu")` に依存していました。この経路はオフライン環境で失敗したため、ローカルにインストール済みの `sacrebleu` パッケージを直接使う実装に変更しました。

これにより、Hugging Face Hub に接続できない環境でも generation 指標を評価できます。

## 精度が低い原因の考察

本セッションでは、論文の GPT-4o mini 実験結果と比較したときに、現在のローカル LoRA 実装で精度が伸びない原因を調査しました。結論として、主要因はモデルの一般ベンチマーク性能そのものよりも、学習形式・推論形式・評価実行環境の差分にある可能性が高いです。

### 1. prompt/input 部分にも loss が乗っていた

以前の学習データは、以下のような 1 本の `text` として `SFTTrainer` に渡していました。

```text
prompt

入力:
input

出力:
target<EOS>
```

この形式では、設定しない限り `prompt` や `input` 部分にも next-token prediction の loss が乗ります。OpenAI の fine-tuning では通常、assistant 出力側が主な学習対象になるため、この差分は大きいと考えられます。

古典籍 OCR 訂正では入力が長く、出力形式も厳密です。そのため、入力側まで学習対象にしてしまうと、モデルが「入力を読んで出力だけを返す」挙動よりも、「長い文字列全体を続きとして生成する」挙動を学びやすくなります。これがタグ崩れ、反復、説明文混入、短い出力への崩壊の一因になっていた可能性があります。

対応として、学習データを `prompt` と `completion` に分離し、既定で `completion_only_loss: true` を使うようにしました。

### 2. Instruct モデルに対して chat template を使っていなかった

論文の GPT-4o mini fine-tuning は user / assistant の会話形式に近い API で実行されていると考えられます。一方、以前のローカル実装では、Qwen3.5 に対してもプレーンテキストの連結形式で学習・推論していました。

この差分により、モデルが事前学習・instruction tuning で期待している会話テンプレートと、実際に与えられる入力形式がずれていた可能性があります。

対応として、`use_chat_template: true` の場合は tokenizer の `apply_chat_template()` を使うようにし、学習時と推論時の形式を揃えました。

### 3. chat template 用データ形式が Unsloth/TRL の期待とずれていた

修正後に `train-detector` を実行したところ、以下のエラーが発生しました。

```text
TypeError: string indices must be integers, not 'str'
```

原因は、Unsloth/TRL 側が chat message の `content` を次のような content parts 形式として処理していたことです。

```json
[{"type": "text", "text": "..."}]
```

当初の実装では `content` に文字列を直接入れていたため、`content["type"]` のようなアクセスで失敗していました。

対応として、学習時の `prompt` / `completion` と、推論時の `apply_chat_template()` 入力を、どちらも content parts 形式に変更しました。

修正後、以下の最小実行が完走することを確認しました。

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner train-detector \
  --processed-dir data/processed \
  --config-path configs/detector.yaml \
  --fold 0 \
  --output-dir results/qwen35_instruct_detector_debug \
  --max-samples 1
```

### 4. detector の既存 smoke 成果物は有効な誤り検出をしていなかった

既存の `results/detector_gpt_oss_20b/fold_0_predictions.smoke.jsonl` は 1 件だけの smoke 予測ですが、評価すると以下でした。

| 指標 | 値 |
|---|---:|
| accuracy | 0.938650 |
| precision | 0.000000 |
| recall | 0.000000 |
| F1 | 0.000000 |

accuracy が高く見えるのは、誤りでない文字のほうが多いクラス不均衡の影響です。precision / recall / F1 が 0 であるため、実質的には誤り span を検出できていません。

したがって、今後は detector の評価では accuracy だけを見ず、precision / recall / F1 を主指標として確認する必要があります。

### 5. 2 段階全文出力は論文再現として維持し、Edit-Only は改善比較に分離する

全文訂正は、未タグ領域まで LLM が再生成して壊すリスクがあります。ただし、論文再現の主実験では `detector` と `corrector` の 2 段階全文出力を維持する必要があります。

そのため、本セッションでは `Edit-Only` を主経路に置き換えるのではなく、論文準拠の 2 段階法とは別の改善比較として扱う方針に整理しました。

## 本セッションでの修正内容

主な修正は以下です。

- `src/koten_refiner/train.py`
  - SFT 学習行を `prompt` / `completion` に分離しました。
  - `completion_only_loss` を設定から切り替えられるようにしました。
  - `use_chat_template` が有効な場合、chat message を content parts 形式で生成するようにしました。
  - legacy の全文 `text` 形式も `completion_only_loss: false` で使えるように残しました。

- `src/koten_refiner/inference.py`
  - 学習済み model dir の `train_config.json` を読み、推論時にも `use_chat_template` を反映するようにしました。
  - chat template 使用時の推論入力を content parts 形式に統一しました。
  - adapter dir の読み込みで Unsloth PEFT モデルを扱う経路を追加しました。

- `src/koten_refiner/metrics.py`
  - `evaluate.load("sacrebleu")` 依存をやめ、ローカル `sacrebleu` パッケージで BLEU を計算するようにしました。

- `configs/*.yaml`
  - Qwen / Gemma / gpt-oss 系設定に `completion_only_loss: true` と `use_chat_template: true` を明示しました。
  - Qwen 系モデル名は `Qwen/Qwen3.5-9B` に統一しました。

- `tests/test_train.py`, `tests/test_inference.py`
  - completion-only 形式、chat content parts 形式、推論時 chat template 利用のテストを追加・更新しました。
  - モデル名の固定検証は避け、設定の構造と学習形式を検証する形にしました。

検証結果:

検証コマンド:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

結果:

```text
52 passed, 2 warnings
```

## 学習の実行状況

新規 LoRA 学習は実行できませんでした。理由は、この実行環境に利用可能な CUDA デバイスがないためです。

Qwen / Gemma / gpt-oss の fine-tuning を CPU で実行するのは、このデータセット規模では現実的ではなく、有意な評価結果も得られないため実施していません。

GPU 環境での推奨実行手順:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner train-detector \
  --processed-dir data/processed \
  --config-path configs/detector.yaml \
  --fold 0 \
  --output-dir results/qwen35_instruct_detector

env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner predict-fold \
  --task detector \
  --model-dir results/qwen35_instruct_detector/fold_0 \
  --processed-dir data/processed \
  --fold 0 \
  --split test \
  --batch-size 4 \
  --output-path results/qwen35_instruct_detector/fold_0_predictions.jsonl

env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner evaluate-predictions \
  --task detector \
  --predictions-path results/qwen35_instruct_detector/fold_0_predictions.jsonl \
  --output-path results/qwen35_instruct_detector/fold_0_metrics.json

env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner train-corrector \
  --processed-dir data/processed \
  --config-path configs/corrector.yaml \
  --task corrector \
  --fold 0 \
  --output-dir results/qwen35_instruct_corrector

env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner prepare-corrector-test \
  --processed-dir data/processed \
  --detector-predictions results/qwen35_instruct_detector/fold_0_predictions.jsonl \
  --fold 0 \
  --output-path results/qwen35_instruct_corrector/fold_0_test_input.jsonl

env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner predict-fold \
  --task corrector \
  --model-dir results/qwen35_instruct_corrector/corrector_fold_0 \
  --input-override results/qwen35_instruct_corrector/fold_0_test_input.jsonl \
  --output-path results/qwen35_instruct_corrector/fold_0_predictions.jsonl

env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner evaluate-predictions \
  --task corrector \
  --predictions-path results/qwen35_instruct_corrector/fold_0_predictions.jsonl \
  --output-path results/qwen35_instruct_corrector/fold_0_metrics.json
```

## 次の作業

`nvidia-smi` が正常に動作し、`torch.cuda.is_available()` が `True` になる環境で上記手順を実行してください。fold 0 が完了した後、fold 1-4 でも同じ手順を実行し、最後に `koten-refiner summarize-metrics` で 5-fold 平均を集計します。

## 2026-05-04 追加評価: Qwen3.5-4B detector

ユーザー指摘に基づき、Qwen3.5 系は QLoRA ではなく通常 LoRA で扱う方針に変更しました。また、モデルを `Qwen/Qwen3.5-4B` に変更し、detector fold 0 を学習しました。

学習済みモデル:

```text
results/qwen35_4b_special_token_detector/fold_0
```

学習設定の要点:

```json
{
  "model": {
    "name": "Qwen/Qwen3.5-4B",
    "load_in_4bit": false
  },
  "train": {
    "completion_only_loss": true,
    "use_chat_template": true,
    "add_error_tag_tokens": true,
    "train_error_tag_embeddings": true
  }
}
```

学習は `1041/1041 step` で正常終了しました。`train_loss` は `2.354`、終盤の logging loss はおおむね `2.17` から `2.26` の範囲でした。

全件 test 予測を開始しましたが、デフォルト設定では 4 件生成時点で約 7 時間規模の見込みになったため中断しました。その後、まず 100 件サンプルで `--max-new-tokens 512` を指定して評価しました。

評価コマンド:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner predict-fold \
  --task detector \
  --model-dir results/qwen35_4b_special_token_detector/fold_0 \
  --processed-dir data/processed \
  --fold 0 \
  --split test \
  --batch-size 4 \
  --max-samples 100 \
  --max-new-tokens 512 \
  --output-path results/qwen35_4b_special_token_detector/fold_0_predictions_100.jsonl

env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner evaluate-predictions \
  --task detector \
  --predictions-path results/qwen35_4b_special_token_detector/fold_0_predictions_100.jsonl \
  --output-path results/qwen35_4b_special_token_detector/fold_0_metrics_100.json

env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner analyze-detector-predictions \
  --predictions-path results/qwen35_4b_special_token_detector/fold_0_predictions_100.jsonl \
  --output-path results/qwen35_4b_special_token_detector/fold_0_diagnostics_100.json
```

100 件サンプルの評価結果:

```json
{
  "accuracy": 0.8101727376523885,
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0
}
```

出力診断:

```json
{
  "total": 100,
  "has_open_tag": 0,
  "has_close_tag": 0,
  "has_error_pair": 0,
  "valid_error_markup": 0,
  "invalid_error_markup": 55,
  "no_error_tags": 100,
  "contains_unknown_markup": 55,
  "contains_thinking_marker": 0,
  "contains_explanatory_text": 2,
  "too_short": 2,
  "too_long": 61,
  "normalized_to_raw_ocr": 100,
  "valid_error_markup_rate": 0.0,
  "normalized_to_raw_ocr_rate": 1.0,
  "explanatory_text_rate": 0.02
}
```

今回の結果では、`<error>` / `</error>` は tokenizer に保存されていますが、生成結果にはタグが一度も現れませんでした。代わりに、多言語の断片、特殊メディアトークン、反復的な数字列などが混入しています。そのため detector の正例予測が 0 件になり、precision / recall / F1 はすべて 0.0 です。

現時点の主な考察:

- 精度低下の直接原因は、評価時に valid な `<error>...</error>` マークアップが生成されていないことです。
- 4B 通常 LoRA 学習は完走しており loss も下がっていますが、detector 形式の出力制御には失敗しています。
- `completion_only_loss` と特殊トークン追加だけでは不十分で、Qwen3.5-4B では出力形式が崩れている可能性があります。
- 100 件中 100 件が raw OCR へ正規化されているため、accuracy 0.81 は「誤りなし」とみなす多数派寄りの見かけの値であり、detector としては機能していません。

次の改善候補:

1. まず train split の少数件で推論し、学習データすら再現できるか確認する。
2. detector を全文生成ではなく、文字単位 BIO / span JSON など構造化された短い出力に変更する。
3. `<error>` タグ方式を続ける場合は、デコード時の禁止トークン・停止条件・繰り返し抑制を追加する。
4. 4B より大きい instruct モデル、または論文条件に近いモデル API で同じプロンプト・同じ評価データを比較する。

## 2026-05-04 追加実験: llm-jp-4-8b-instruct detector

日本語に強いモデルとして `llm-jp/llm-jp-4-8b-instruct` を使い、detector fold 0 を LoRA 学習しました。Hugging Face の `alfredplpl/llm-jp-4-8b-instruct-zundamon-lora` は LoRA アダプタ例であり、今回のベースモデルには `llm-jp/llm-jp-4-8b-instruct` を使用しました。

主な実装修正:

- llm-jp4 の chat template は `content` に文字列を要求するため、`train.chat_content_parts: false` を追加し、Qwen-VL 形式の `{"type": "text"}` 配列を使わない経路を実装しました。
- llm-jp4 の生成ヘッダ `<|channel|> final<|message|>` と終端 `<|end|>` を推論後に除去する処理を追加しました。
- llm-jp4 では `<error>` が未知語扱いになるため、特殊トークン追加は無効化しました。
- 生成で頻発した `</エラー>`、`</ error>`、`</errors>`、`<Error>`、属性付き `<error ...>` を `<error>` / `</error>` に正規化する後処理を追加しました。

追加設定ファイル:

```text
configs/llmjp4_detector_smoke.yaml
configs/llmjp4_detector.yaml
```

本学習コマンド:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner train-detector \
  --processed-dir data/processed \
  --config-path configs/llmjp4_detector.yaml \
  --fold 0 \
  --output-dir results/llmjp4_detector
```

学習結果:

```text
model_dir: results/llmjp4_detector/fold_0
steps: 1041/1041
epoch: 3
train_loss: 0.1587
train_runtime: 6735 sec
wandb: https://wandb.ai/kotomiya07/koten-text-refiner/runs/5amfc6sb
```

train 10 件での確認では、タグ正規化前は valid markup が `1/10` でした。正規化後は `7/10` まで改善しました。

train 10 件、タグ正規化後:

```json
{
  "accuracy": 0.828551912568306,
  "precision": 0.46774193548387094,
  "recall": 0.05823293172690763,
  "f1": 0.10357142857142858
}
```

診断:

```json
{
  "total": 10,
  "has_open_tag": 10,
  "has_close_tag": 10,
  "has_error_pair": 10,
  "valid_error_markup": 7,
  "invalid_error_markup": 3,
  "no_error_tags": 0,
  "contains_unknown_markup": 3,
  "contains_thinking_marker": 0,
  "contains_explanatory_text": 0,
  "too_short": 0,
  "too_long": 10,
  "normalized_to_raw_ocr": 3,
  "valid_error_markup_rate": 0.7,
  "normalized_to_raw_ocr_rate": 0.3,
  "explanatory_text_rate": 0.0
}
```

test 100 件評価:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run koten-refiner predict-fold \
  --task detector \
  --model-dir results/llmjp4_detector/fold_0 \
  --processed-dir data/processed \
  --fold 0 \
  --split test \
  --batch-size 1 \
  --max-samples 100 \
  --max-new-tokens 768 \
  --output-path results/llmjp4_detector/fold_0_predictions_test_100.jsonl
```

```json
{
  "accuracy": 0.8121247836175464,
  "precision": 0.5234305923961097,
  "recall": 0.11486224291812185,
  "f1": 0.18838504375497217
}
```

診断:

```json
{
  "total": 100,
  "has_open_tag": 100,
  "has_close_tag": 100,
  "has_error_pair": 100,
  "valid_error_markup": 55,
  "invalid_error_markup": 45,
  "no_error_tags": 0,
  "contains_unknown_markup": 45,
  "contains_thinking_marker": 0,
  "contains_explanatory_text": 2,
  "too_short": 0,
  "too_long": 100,
  "normalized_to_raw_ocr": 45,
  "valid_error_markup_rate": 0.55,
  "normalized_to_raw_ocr_rate": 0.45,
  "explanatory_text_rate": 0.02
}
```

考察:

- Qwen3.5-4B の test 100 件では `<error>` タグが一度も有効に出ず F1 は `0.0` でしたが、llm-jp4 では全件でタグ生成が発生し、test 100 件 F1 は `0.188` まで改善しました。モデル選択の影響は明確にあります。
- 一方で、train 10 件でも recall が `0.058` に留まるため、まだ学習データの再現に失敗しています。汎化以前に detector の出力形式とスパン選択が安定していません。
- 主要な失敗は「タグを出さない」から「タグは出すが、過剰・過少・構文崩れを起こす」に変わりました。特に `too_long: 100/100` で、全文再生成方式が長文ページに対して不安定です。
- タグ表記揺れの正規化は有効でしたが、これは構文崩れの一部を救うだけです。スパン境界の誤りや過剰検出は解決していません。

次の改善案:

1. detector を全文タグ付き再生成から、短い span JSON、文字 offset、または BIO ラベル予測に変更する。
2. `<error>` タグ方式を続ける場合は、XML/HTML 風タグの制約付きデコード、または生成後に OCR 原文へ再アラインしてタグだけを投影する後処理を入れる。
3. 長文をページ全体で処理せず、行単位または短いチャンク単位で detector を学習・推論する。
4. train split の少数件で高 recall を達成できるまで、test 評価よりも学習データ再現性を優先してデバッグする。
