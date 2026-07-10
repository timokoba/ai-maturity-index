"""FinBERT-FLS forward-looking scoring with on-disk caching.

Model: `yiyanghkust/finbert-fls` (Huang, Wang, Yang 2023, CAR), a 3-class
classifier (specific / non-specific / not forward-looking) trained on 10-K and
10-Q management-discussion sentences. Used for the Governance dimension (Item 1A
risk factors), where FinBERT-tone sentiment is uninformative -- risk factors are
uniformly negative -- and the signal of interest is instead how prospectively a
firm frames its AI risk.

The per-sentence class probabilities are returned (not just the argmax label) so
downstream aggregation can grade the *degree* of forward-looking framing rather
than count hard labels. Each sentence's probabilities are cached by
`sha256(sentence)` in a directory separate from the sentiment cache.
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

MODEL_NAME = "yiyanghkust/finbert-fls"
CACHE_DIR = DATA_CACHE / "edgar" / "_fls_cache"
_CLASSES = ("specific", "nonspecific", "not")
COLUMNS = ["sentence", "p_specific", "p_nonspecific", "p_not", "label", "confidence"]


def _hash(sentence: str) -> str:
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()


def _cache_path(sentence_hash: str) -> Path:
    return CACHE_DIR / sentence_hash[:2] / f"{sentence_hash}.json"


@lru_cache(maxsize=1)
def _load_pipeline():
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer, pipeline

    # See sentiment.py: FinBERT's 2020-era config needs the explicit BERT classes
    # under current transformers (Auto* requires model_type / fast-tokenizer files).
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification", model=model, tokenizer=tokenizer,
        device=device, top_k=None, truncation=True, max_length=256,
    )


def _read_cached(sentences: list[str]) -> tuple[dict[int, dict[str, float]], list[int]]:
    cached: dict[int, dict[str, float]] = {}
    misses: list[int] = []
    for i, sent in enumerate(sentences):
        path = _cache_path(_hash(sent))
        try:
            cached[i] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001  (missing or unreadable -> rescore)
            misses.append(i)
    return cached, misses


def _write_cache(sentence: str, scores: dict[str, float]) -> None:
    path = _cache_path(_hash(sentence))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores), encoding="utf-8")


def _normalize_label(label: str) -> str:
    """Map a raw FinBERT-FLS label ("Specific FLS" / "Non-specific FLS" /
    "Not-FLS") to specific / nonspecific / not. Order matters: "Non-specific"
    also contains "specific", so "not" and "non-specific" are tested first."""
    s = label.strip().lower()
    if "not" in s:
        return "not"
    if "non-specific" in s or "non specific" in s:
        return "nonspecific"
    return "specific" if "specific" in s else "not"


def score_forward_looking(sentences: list[str], batch_size: int = 32) -> pd.DataFrame:
    """Score sentences with FinBERT-FLS.

    Returns one row per sentence with the class probabilities `p_specific`,
    `p_nonspecific`, `p_not`, the argmax `label`, and its `confidence`. Cached
    sentences skip the model entirely.
    """
    if not sentences:
        return pd.DataFrame(columns=COLUMNS)

    cached, missing_idx = _read_cached(sentences)
    results: dict[int, dict[str, float]] = dict(cached)

    if missing_idx:
        pipe = _load_pipeline()
        to_score = [sentences[i] for i in missing_idx]
        log.info("Scoring %d uncached sentences (FinBERT-FLS)", len(to_score))
        for start in range(0, len(to_score), batch_size):
            batch = to_score[start : start + batch_size]
            for j, pred in enumerate(pipe(batch)):
                scores = {c: 0.0 for c in _CLASSES}
                for entry in (pred if isinstance(pred, list) else [pred]):
                    scores[_normalize_label(entry["label"])] = float(entry["score"])
                _write_cache(batch[j], scores)
                results[missing_idx[start + j]] = scores

    rows: list[dict] = []
    for i, sent in enumerate(sentences):
        s = results.get(i, {"specific": 0.0, "nonspecific": 0.0, "not": 1.0})
        top = max(_CLASSES, key=lambda c: s[c])
        label = {"specific": "specific_fls", "nonspecific": "nonspecific_fls", "not": "not_fls"}[top]
        rows.append(dict(
            sentence=sent, p_specific=s["specific"], p_nonspecific=s["nonspecific"],
            p_not=s["not"], label=label, confidence=s[top],
        ))
    return pd.DataFrame(rows, columns=COLUMNS)
