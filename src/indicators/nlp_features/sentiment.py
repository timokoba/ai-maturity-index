"""FinBERT-tone sentiment scoring with on-disk caching.

Model: `yiyanghkust/finbert-tone` (Huang, Wang, Yang 2023, CAR). 3-class
output: positive / neutral / negative. Trained on 10-K, analyst-report,
and earnings-call text; the academic default for finance NLP.

Each sentence's logits are cached by `sha256(sentence)` so re-runs of
the pipeline on overlapping data sets are idempotent and fast.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..common.io import DATA_CACHE

log = logging.getLogger(__name__)

MODEL_NAME = "yiyanghkust/finbert-tone"
CACHE_DIR = DATA_CACHE / "edgar" / "_sentiment_cache"


def _hash(sentence: str) -> str:
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()


def _cache_path(sentence_hash: str) -> Path:
    return CACHE_DIR / sentence_hash[:2] / f"{sentence_hash}.json"


@lru_cache(maxsize=1)
def _load_pipeline():
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer, pipeline

    # FinBERT is fine-tuned bert-base with a custom 30,873-token vocab. Its 2020-era
    # config lacks the `model_type`/fast-tokenizer files that transformers' Auto*
    # classes now require, so we load the BERT classes explicitly (same weights and
    # WordPiece tokenization, no conversion needed).
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True,
        max_length=256,
    )


def _read_cached(sentences: list[str]) -> tuple[dict[int, dict[str, float]], list[int]]:
    cached: dict[int, dict[str, float]] = {}
    misses: list[int] = []
    for i, sent in enumerate(sentences):
        path = _cache_path(_hash(sent))
        if path.exists():
            try:
                cached[i] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                misses.append(i)
        else:
            misses.append(i)
    return cached, misses


def _write_cache(sentence: str, scores: dict[str, float]) -> None:
    h = _hash(sentence)
    path = _cache_path(h)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores), encoding="utf-8")


def _normalize_label(label: str) -> str:
    s = label.strip().lower()
    if s.startswith("pos"):
        return "positive"
    if s.startswith("neg"):
        return "negative"
    return "neutral"


def score_sentences(
    sentences: list[str],
    batch_size: int = 32,
) -> pd.DataFrame:
    """Score a list of sentences with FinBERT-tone.

    Returns a dataframe with columns `sentence`, `pos`, `neu`, `neg`,
    `label`, `confidence`. Cache hits avoid model invocation entirely.
    """
    if not sentences:
        return pd.DataFrame(columns=["sentence", "pos", "neu", "neg", "label", "confidence"])

    cached, missing_idx = _read_cached(sentences)
    results: dict[int, dict[str, float]] = dict(cached)

    if missing_idx:
        pipe = _load_pipeline()
        to_score = [sentences[i] for i in missing_idx]
        log.info("Scoring %d uncached sentences", len(to_score))
        for batch_start in range(0, len(to_score), batch_size):
            batch = to_score[batch_start : batch_start + batch_size]
            preds = pipe(batch)
            for j, (sent, pred) in enumerate(zip(batch, preds)):
                scores = {"pos": 0.0, "neu": 0.0, "neg": 0.0}
                items = pred if isinstance(pred, list) else [pred]
                for entry in items:
                    label = _normalize_label(entry["label"])
                    key = {"positive": "pos", "neutral": "neu", "negative": "neg"}[label]
                    scores[key] = float(entry["score"])
                _write_cache(sent, scores)
                results[missing_idx[batch_start + j]] = scores

    rows: list[dict] = []
    for i, sent in enumerate(sentences):
        s = results.get(i, {"pos": 0.0, "neu": 1.0, "neg": 0.0})
        label_idx = max(("pos", "neu", "neg"), key=lambda k: s[k])
        label = {"pos": "positive", "neu": "neutral", "neg": "negative"}[label_idx]
        rows.append(
            dict(
                sentence=sent,
                pos=s["pos"],
                neu=s["neu"],
                neg=s["neg"],
                label=label,
                confidence=s[label_idx],
            )
        )
    return pd.DataFrame(rows)
