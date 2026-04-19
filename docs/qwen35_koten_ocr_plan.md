# Qwen3.5-9B + Unsloth による古典籍 OCR 誤り訂正実験計画

## Summary

- 論文の本体である 2 段階法を、手元データに合わせてできるだけ同一手順で再現する。
- 論文のデータ前提は `A がある作品は A を採用し、A がなく B のみある作品は B を採用、A/B 両方ある 10 作品は A を採用、さらに対応確認が取れなかった 4 作品を除外して 122 作品` である。
- ローカル `datasets` には `NDL=126作品`, `humanA=80作品`, `humanB=56作品`, `A∩B=10作品` があり、`A優先 + B-only 採用` で 126 作品が構成できる。
- 本実験では、論文公開時点から 4 作品増えた前提で、`A優先 + B-only 採用` の 126 作品を正式対象とする。
- 使用モデルは論文の `GPT-4o mini` ではなく `Qwen/Qwen3.5-9B` に置き換え、学習は `Unsloth` の bf16 LoRA で行う。
- 小学館コーパスが手元にないため、論文 5.3 / 6.3 の追加学習比較は再現対象から外し、まず 5.1 / 5.2 / 6.1 / 6.2 の本体実験を再現する。
- 論文再現後に、過剰再生成を抑える改善法として `Edit-Only Correction` を追加比較する。

## Paper-Aligned Reproduction

### Data construction

- 入力 OCR は `datasets/ndl/<work>/json/*.json` を使う。
- 正解は作品ごとに `humanA` を優先し、`humanA` が存在しない作品のみ `humanB` を使う。
- 具体的には `selected_human_source(work) = humanA if work in humanA else humanB` とする。
- A/B 両方ある 10 作品では常に `humanA` を使う。これは論文の記述と一致させる。
- ペアリングは `(work_id, page_number:int)` で行う。
- `ndl` 側は `<work>-<page5>.json` から `page_number = int(<page5>)` を得る。
- `humanA` 側は `<work>_<page4>_qc.txt` から `page_number = int(<page4>)` を得る。
- `humanB` 側は `<work>_<page4>_qc.text` から `page_number = int(<page4>)` を得る。
- 突き合わせは `(work_id, page_number)` の完全一致に限定し、桁数や区切り文字の差は `int` 化で吸収する。
- OCR 文字列は `ndl` JSON の `contents[*][-1]` を保存順のまま改行連結して復元する。
- 人手転記側も `humanA/json` または `humanB/json` の保存順が `text` とほぼ一致することを確認したため、ページ内の直列化規則は `保存順` を正規とみなす。
- 単純な座標再ソートは採用しない。検証では `x降順, y昇順` の幾何ソートをかけると `human/json -> human/text` の一致が崩れるページがあり、`NDL` 側でも一貫して改善しなかった。
- したがって NDL と human のページごとのデータ生成では、どちらも同じ「保存順で改行連結」の規則を使う。
- 正解文字列は選択された人手転記ソースの `text` を使い、改行コードは `CRLF -> LF` に正規化したうえで末尾空白を除去する。
- `humanA/json` は参照用の座標データとしてのみ扱い、学習・評価入力には使わない。
- `humanB/json` も同様に参照用であり、学習・評価入力には使わない。
- ローカル再現の正式母集団は `A優先 + B-only 採用` の 126 作品とする。
- OCR または正解が空のページは除外する。論文に明記はないが、空入力・空正解は誤り訂正タスクを成立させないため除外を正式仕様とする。

### Split policy

- 論文 5.1 / 5.2 に合わせ、データセットは `(学習:テスト) = (4:1)` の 5 分割交差検定とする。
- 分割単位はページ単位にする。前回案の `GroupKFold(work_id)` は論文からの逸脱なので再現実験では採用しない。
- fold 分割は `KFold(n_splits=5, shuffle=True, random_state=42)` 相当で固定し、生成された fold ID を全条件で共有する。
- 誤り検出、1 段階訂正、2 段階訂正はすべて同じ 5-fold 分割を使う。
- 2 段階訂正では、論文記載どおり「誤り検出と同様のデータの分割方法で、問題が被らないように 5 分割」する。実装上は共通 fold ID を保存して全条件で再利用する。
- 論文に validation の記載がないため、再現実験では early stopping を入れず固定 epoch 学習にする。checkpoint は各 epoch 保存し、最終 epoch を採用する。

### Detector setup

- 学習データは OCR と正解の文字列比較から作る。
- アラインメントは文字単位 Levenshtein opcodes で作る。
- `equal` 以外の `replace / delete / insert` を OCR 側 span に射影し、連続または隣接する誤り操作は 1 つの `<error>...</error>` span にまとめる。
- `insert` だけで OCR 側に長さ 0 の場合は、その挿入位置の直前文字を含む最小 1 文字 span に拡張してタグ付けする。
- ただし文頭挿入で直前文字が存在しない場合は、直後文字を含む最小 1 文字 span に拡張する。
- 教師出力は「OCR 文中の誤り部分のみを `<error>...</error>` で囲んだ文字列」とする。
- 入力は OCR 結果、出力はタグ付き OCR 結果。
- 論文の検出プロンプトはそのまま使う。

```text
次の文章は OCR の出力結果です。文字認識が誤っている部分にのみ<error>タグ</error>を付けてください.

注意:
・誤りのある語句だけを<error>タグ</error>で囲むこと。
・それ以外の補足説明、前置き、警告などは一切出力しないこと。
・タグ付きの文章だけを返してください。
```

- 評価指標は `Accuracy / Precision / Recall / F1`。
- 評価単位は論文どおりサブトークンベースとする。
- トークナイズは日本語 RoBERTa tokenizer を使う。論文は「RoBERTa のトークナイザー」とだけ書いており厳密なモデル名はないため、実装では `rinna/japanese-roberta-base` を採用する。
- サブトークンラベルは、OCR 文字列側の各サブトークンが正解タグ span に重なれば `1`、重ならなければ `0` とする。

### Corrector setup

- 論文どおり、訂正モデルの入力は `error タグ付き text` と `タグなし text` を `<sep>` で連結した形式にする。
- 学習時は detector の予測ではなく oracle error tag を使う。
- 評価時とテスト時は detector の予測結果に基づく error tag を使う。
- 出力は訂正後の全文とする。
- 論文の訂正プロンプトはそのまま使う。

```text
次の文章は手書き文字の OCR 結果です。<error>タグで囲まれた箇所に OCR による誤りがあります。
<error>タグで囲まれた部分のみを訂正し，本来書かれていた通りの文字列に戻してください。
入力は，<error>タグ付きの文と，タグなしの文が<sep>で連結された形式です。
```

- 1 段階ベースラインは detector を使わず、`raw_ocr -> 正解全文` のみで学習する。
- 1 段階ベースラインのプロンプトは訂正プロンプトから `<error>` と `<sep>` の説明を外し、「次の文章は手書き文字の OCR 結果です。本来書かれていた通りの文字列に戻してください。」を基本文面とする。
- 評価指標は論文どおり `BLEU / CRR / WRR` に固定する。
- BLEU は `evaluate` の `sacrebleu` を使う。
- `CRR = 1 - CER`、`WRR = 1 - WER` とする。
- WER 計算の分かち書きは、論文 5.3 でも MeCab を使っているため、再現実装でも `MeCab + UniDic` に固定する。

### Model and training adaptation

- 論文の LLM は `gpt-4o-mini-2024-07-18` だが、再現実装では `Qwen/Qwen3.5-9B` に置き換える。
- 学習フレームワークは `Unsloth` を使う。
- 依存は `uv` で管理し、`unsloth` が解決する `transformers` 互換版に従う。`transformers>=5` のような独自固定は行わない。
- CLI は `prepare-data`, `train-detector`, `train-corrector`, `eval-cv`, `run-improvement` の 5 本を用意する。
- 学習は bf16 LoRA に固定し、4-bit QLoRA は初回再現では使わない。
- 初回既定値は `max_seq_length=4096`, `per_device_batch_size=1`, `gradient_accumulation_steps=16`, `lr=1e-4`, `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `epochs=3`, `seed=42` とする。
- これは論文の非公開学習設定を Qwen3.5-9B + RTX 4090 24GB 上で実行可能にするための置換設定であり、論文との差分として明示して記録する。

## Improvement Experiment

- 再現ベースラインの後に `Edit-Only Correction` を比較する。
- 目的は、論文 7 章で指摘された「短い出力に崩れる」「全文を再生成して壊す」失敗を抑えることにある。
- 入力では各誤り span に連番を付ける。例: `今日は<error id="1">天期</error>がよい。`
- 出力は `span_id<TAB>corrected_text` の列挙に固定する。変更不要な誤検出 span は `span_id<TAB><KEEP>` を出力させる。
- 復元器は `<KEEP>` を元の OCR span に戻し、それ以外のみを差し替える。未タグ領域は常に元文を保持する。
- 改善比較は `訂正前`, `1段階訂正`, `論文準拠 2 段階`, `Edit-Only 2 段階` の 4 条件にする。
- `Noise-Aware Curriculum` は初回計画から外す。論文再現との差が増えすぎるため、まずは `Edit-Only` 単独の効果を測る。

## Outputs and Logging

- 各 fold の保存先は `results/<run_name>/fold_{k}/` とする。
- 保存物は `metrics.json`, `predictions.jsonl`, `config.yaml`, `train_log.jsonl` を標準とする。
- fold 横断の集計は `results/<run_name>/summary.json` と `summary.csv` に保存する。
- run ごとに `paper_aligned=true/false` を設定し、論文準拠実験と改善実験を明確に分ける。
- ログ基盤は最初はローカルファイルのみとし、`wandb` は入れない。

## Test Plan

- ページ番号正規化テスト: `200003425-00001.json` と `200003425_0001_qc.txt` が同じ `(work_id, page_number)` に変換されることを確認する。
- OCR 復元テスト: 空 `contents`、1 要素、複数要素で期待どおりの復元が行われることを確認する。
- 順序妥当性テスト: `human/json` を保存順で連結した文字列が `human/text` と一致または高一致になることをサンプルページで確認し、幾何ソートより悪化しないことを検証する。
- 検出教師生成テスト: OCR と正解が一致する場合はタグなし、局所誤りでは誤り部分だけがタグ化されることを確認する。
- 検出教師生成テストには、文頭 insert、文中 insert、連続 replace、delete を含むエッジケースを入れる。
- 検出評価テスト: サブトークンラベルが正しく 0/1 化され、F1 が既知の小例で一致することを確認する。
- 訂正入力生成テスト: `tagged_ocr<sep>raw_ocr` が論文形式どおりに生成されることを確認する。
- `Edit-Only` 復元テスト: `<KEEP>`、複数 span、未タグ領域不変を確認する。
- 指標テスト: BLEU/CER/WER/CRR/WRR が既知の例で一致することを確認する。
- 再現性テスト: seed=42 で同一 fold 定義を再読込した場合に分割が不変であることを確認する。
- 学習スモークテスト: 開発用に 10 作品サブセットを使って detector/corrector が完走することを確認する。ただし正式結果は 126 作品 5-fold 条件でのみ報告する。

## Acceptance Criteria

- 論文準拠 2 段階法が、対象コーパスの 5-fold 平均で `1段階訂正` より BLEU / CRR / WRR のいずれかを改善する。
- `Edit-Only 2 段階` が `論文準拠 2 段階` と比べて BLEU を改善するか、少なくとも CRR/WRR を落とさず短文崩壊件数を減らす。
- 全 run が fold 別予測・fold 別指標・集計結果を `results/` に保存する。

## Assumptions

- 論文の厳密な学習 API は再現できないため、論文との差分は `LLM が GPT-4o mini ではなく Qwen3.5-9Bであること` と `学習実装が Unsloth LoRA であること` の 2 点を主要差分として扱う。
- 論文データ前提としては `humanA のみ` ではなく `A優先 + B-only 採用` が正しい。前回の 80 作品前提は破棄する。
- 本実験では、論文時点から 4 作品増えた前提を採用し、126 作品を正式対象とする。
- 論文の追加学習比較は、コーパス不在のため今回の再現対象から外す。
- 論文に記載のない部分は、論文本体を崩さない最小限の実装判断だけを入れる。

## References

- Qwen3.5-9B model card: <https://huggingface.co/Qwen/Qwen3.5-9B>
- Unsloth Qwen3 fine-tuning docs: <https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune>
- Unsloth requirements: <https://docs.unsloth.ai/get-started/fine-tuning-for-beginners/unsloth-requirements>
