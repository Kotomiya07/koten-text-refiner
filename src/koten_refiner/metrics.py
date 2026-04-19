from __future__ import annotations

import MeCab
from jiwer import cer, wer


_MECAB = MeCab.Tagger()
_BLEU = None


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def tokenize_ja(text: str) -> str:
    parsed = _MECAB.parse(text)
    if not parsed:
        return text
    tokens = [line.split("\t", 1)[0] for line in parsed.splitlines() if line and line != "EOS"]
    return " ".join(tokens)


def compute_text_metrics(prediction: str, reference: str) -> dict[str, float]:
    global _BLEU
    if _BLEU is None:
        import evaluate

        _BLEU = evaluate.load("sacrebleu")
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    bleu = _BLEU.compute(predictions=[pred], references=[[ref]])["score"]
    crr = 1.0 - cer(ref, pred)
    wrr = 1.0 - wer(tokenize_ja(ref), tokenize_ja(pred))
    return {
        "bleu": float(bleu),
        "crr": float(crr),
        "wrr": float(wrr),
    }
