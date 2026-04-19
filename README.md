# koten-text-refiner

古典籍 OCR 誤り訂正の再現実験用リポジトリです。論文「LLM を利用した古典籍における文字認識誤り訂正」をベースに、`Qwen/Qwen3.5-9B` と `Unsloth` を使って 2 段階法を再現し、その後に `Edit-Only` 改善法を比較できるようにしています。

詳細な実験方針は [docs/qwen35_koten_ocr_plan.md](docs/qwen35_koten_ocr_plan.md) を参照してください。

## Current Status

- `uv` 管理の Python プロジェクトを作成済み
- `NDL + humanA/humanB` からページ単位データを生成する前処理を実装済み
- 5-fold CV の fold 定義と task 別 JSONL 出力を実装済み
- `detector`, `corrector`, `one_stage`, `edit_only` の学習用 CLI を実装済み
- fold 推論、detector 出力から corrector test 入力を作る処理、指標計算、fold 集計を実装済み
- detector 評価はメモリ安全な文字単位を既定とし、`rinna/japanese-roberta-base` の `spiece.model` を直接使うサブトークン評価も明示指定で利用可能
- detector 評価は generation 指標モジュールから分離し、専用の軽量経路で処理するよう整理済み
- `PLAN` の基礎テスト項目に対応する単体テストを追加済み
- 軽量コマンドでのメモリ消費を抑えるため、training / tokenizer 依存は遅延 import に整理済み
- detector の smoke 学習・smoke 推論・文字単位評価・サブトークン評価は `results/smoke_detector_retry/` に成果物あり
- corrector の smoke 用 train JSONL は `results/smoke_corrector/` と `results/smoke_corrector_retry/` にあるが、学習・推論・評価の完走成果物はまだ揃っていない
- `one_stage` と `edit_only` の smoke 完走成果物はまだ記録していない

## Layout

- `src/koten_refiner/`
  - `cli.py`: 実験 CLI
  - `dataset_builder.py`: データ収集、fold 生成、task 別データ作成
  - `alignment.py`: OCR と正解の差分から `<error>` span を作る処理
  - `detector_evaluation.py`: detector 指標専用の軽量評価処理
  - `train.py`: Unsloth を使った SFT 学習入口
  - `inference.py`: fold 推論、`Edit-Only` 復元
  - `evaluation.py`: generation 指標と共通入出力ユーティリティ
- `configs/`
  - `detector.yaml`, `corrector.yaml`, `edit_only.yaml`: 本番寄り設定
  - `detector_smoke.yaml`, `corrector_smoke.yaml`: smoke 用の軽量設定
- `data/processed/`
  - 前処理済み JSONL
- `results/`
  - 学習・推論・評価出力

## Environment

依存インストール:

```bash
uv sync
uv add flash-attn --no-build-isolation
```

CLI ヘルプ:

```bash
uv run koten-refiner --help
```

## Data Preparation

前処理を実行して、ページ単位データと fold ごとの task 行を生成します。

```bash
uv run koten-refiner prepare-data \
  --dataset-dir datasets \
  --output-dir data/processed
```

fold の件数確認:

```bash
uv run koten-refiner eval-cv \
  --processed-dir data/processed
```

小さい fold サブセットを書き出す:

```bash
uv run koten-refiner export-fold \
  --task detector \
  --fold 0 \
  --split train \
  --output-path results/tmp/detector_fold0_small.jsonl \
  --max-samples 8
```

## Training

detector 学習:

```bash
uv run koten-refiner train-detector \
  --processed-dir data/processed \
  --config-path configs/detector.yaml \
  --fold 0 \
  --output-dir results/detector
```

corrector / one-stage 学習:

```bash
uv run koten-refiner train-corrector \
  --processed-dir data/processed \
  --config-path configs/corrector.yaml \
  --task corrector \
  --fold 0 \
  --output-dir results/corrector
```

`Edit-Only` 学習:

```bash
uv run koten-refiner run-improvement \
  --processed-dir data/processed \
  --config-path configs/edit_only.yaml \
  --fold 0 \
  --output-dir results/improvement
```

smoke 実行:

```bash
uv run koten-refiner train-detector \
  --processed-dir data/processed \
  --config-path configs/detector_smoke.yaml \
  --fold 0 \
  --output-dir results/smoke_detector \
  --max-samples 8
```

corrector smoke 用の軽量設定:

```bash
uv run koten-refiner train-corrector \
  --processed-dir data/processed \
  --config-path configs/corrector_smoke.yaml \
  --task corrector \
  --fold 0 \
  --output-dir results/smoke_corrector \
  --max-samples 8
```

## Prediction And Evaluation

detector の fold 推論:

```bash
uv run koten-refiner predict-fold \
  --task detector \
  --model-dir results/detector/fold_0 \
  --processed-dir data/processed \
  --fold 0 \
  --split test \
  --output-path results/detector/fold_0_predictions.jsonl
```

detector 出力から corrector test 入力を生成:

```bash
uv run koten-refiner prepare-corrector-test \
  --processed-dir data/processed \
  --detector-predictions results/detector/fold_0_predictions.jsonl \
  --fold 0 \
  --output-path results/corrector/fold_0_test_input.jsonl
```

corrector の test 推論:

```bash
uv run koten-refiner predict-fold \
  --task corrector \
  --model-dir results/corrector/corrector_fold_0 \
  --input-override results/corrector/fold_0_test_input.jsonl \
  --output-path results/corrector/fold_0_predictions.jsonl
```

指標評価:

```bash
uv run koten-refiner evaluate-predictions \
  --task detector \
  --predictions-path results/detector/fold_0_predictions.jsonl \
  --output-path results/detector/fold_0_metrics.json
```

detector を論文寄りのサブトークン評価で動かす場合:

```bash
uv run koten-refiner evaluate-predictions \
  --task detector \
  --predictions-path results/detector/fold_0_predictions.jsonl \
  --output-path results/detector/fold_0_metrics.json \
  --detector-eval-unit subtoken \
  --detector-tokenizer-model rinna/japanese-roberta-base \
  --detector-char-chunk-size 256
```

corrector / one-stage / edit-only の指標評価:

```bash
uv run koten-refiner evaluate-predictions \
  --task corrector \
  --predictions-path results/corrector/fold_0_predictions.jsonl \
  --output-path results/corrector/fold_0_metrics.json
```

fold 集計:

```bash
uv run koten-refiner summarize-metrics \
  results/corrector/fold_0_metrics.json \
  results/corrector/fold_1_metrics.json \
  results/corrector/fold_2_metrics.json \
  results/corrector/fold_3_metrics.json \
  results/corrector/fold_4_metrics.json \
  --output-path results/corrector/summary.json
```

このコマンドは `summary.json` と同じ場所に `summary.csv` も出力します。

## Smoke Status

- detector smoke の確認済み成果物:
  `results/smoke_detector_retry/fold_0/` に学習済み adapter、`fold_0_test_predictions.jsonl` に推論結果、`fold_0_metrics.json` と `fold_0_subtoken_metrics.json` に評価結果がある
- corrector smoke の確認済み成果物:
  `corrector_fold0_train.jsonl` まではあるが、学習済み model dir や推論・評価ファイルはまだ確認できていない
- one-stage / edit-only smoke:
  現時点では README に載せられる完走成果物はない

## Tests

```bash
uv run pytest -q
```

## Notes

- 現在の前処理では、`NDL` と `human` のページ内順序はどちらも保存順を正規とみなします。単純な座標再ソートは採用していません。
- fold 分割は論文準拠でページ単位 `KFold(n_splits=5, shuffle=True, random_state=42)` です。
- 2 段階 corrector の学習時入力は oracle tag、評価時入力は detector 予測 tag を使います。
- `Edit-Only` の教師はまだ改善余地があります。現時点では局所置換対象を Levenshtein opcodes から近似的に取り出しています。
- detector 評価の既定は文字単位です。これは現環境で最もメモリ安全な経路です。
- detector のサブトークン評価は `--detector-eval-unit subtoken` を明示した場合のみ有効です。`rinna/japanese-roberta-base` の `spiece.model` を `sentencepiece` で直接読み、予測テキストが正解の平文と一致しない場合は、そのサンプルの予測ラベルを全 0 とみなします。
- detector 予測に閉じ忘れた `<error>` タグが含まれる場合は、壊れたタグ列を平文扱いに倒して fail-closed に処理します。これにより評価時の暴走を避けます。
- detector のサブトークン評価はまだメモリ監視下で使う前提です。必要なら `evaluate-predictions --detector-char-chunk-size 128` のように小さくできます。
- detector の active path は `src/koten_refiner/detector_evaluation.py` で、JSONL は行単位で読みます。少なくとも detector 評価の本流では `list(iter_jsonl(...))` を使わない構成にしてあります。
- `prepare-data`, `eval-cv`, `summarize-metrics`, `inspect-record` などの軽量コマンドでは、`unsloth` や `datasets` を先に import しない構成にしてあります。
- `Qwen3.5-9B` の full 実験に入る前に、smoke training の安定化と detector fold 0 の end-to-end 動作確認を進める想定です。
