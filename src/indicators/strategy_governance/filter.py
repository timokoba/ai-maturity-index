"""AI relevance filtering at the sentence level.

The keyword list is a concept-level AI dictionary intended for use on
10-K corporate disclosures. It takes Basnet (2025, "Analyzing the
market's reaction to AI narratives in corporate filings") as its
canonical base, with three Basnet terms removed for precision in the
10-K context ("bot/bots", "cognitive science", "data science"), and
extends it with modern terminology that has become standard in
disclosures since the generative-AI inflection of 2022-2023.

Design principles:

1. Concept-level, not product-level. No vendor names (OpenAI,
   Anthropic, Hugging Face) and no specific product versions
   (ChatGPT, GPT-4o, etc.). Academic dictionaries measure constructs,
   not brand mentions; this mirrors Babina, Fedyk, He, Hodson (2024),
   Cao et al. (2023), and Eisfeldt et al. (2024).

2. Bare "ai" with word-boundary enforcement is included as a single
   token, subsuming every "AI-driven", "AI/ML", "AI agent", "AI
   platform" compound without enumerating them. The regex requires
   non-alphanumeric characters on both sides, so "ai" does not match
   inside "air", "aim", "claim", or "available".

3. Short, ambiguous abbreviations that have non-AI meanings are
   omitted: "ml" (millilitre in pharma filings), "llm" singular (Master
   of Laws). Plurals and disambiguated forms are kept where unambiguous
   ("llms", "nlp").

Matching is case-insensitive.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import pandas as pd


AI_KEYWORDS: tuple[str, ...] = (
    # Bare token (covers all "AI X" compounds via word boundaries)
    "ai",
    # Core concept terms (Basnet 2025 base, three high-FP terms removed)
    "artificial intelligence",
    "machine intelligence",
    "artificial neural network",
    "artificial neural networks",
    "neural network",
    "neural networks",
    "deep learning",
    "machine learning",
    "robotic process automation",
    "intelligent automation",
    "adaptive algorithm",
    "adaptive algorithms",
    "automated decision-making",
    "automated decision making",
    # Learning paradigms
    "reinforcement learning",
    "supervised learning",
    "unsupervised learning",
    "self-supervised learning",
    "transfer learning",
    "federated learning",
    "fine-tuning",
    "fine tuning",
    # Generative AI
    "generative ai",
    "generative artificial intelligence",
    "generative model",
    "generative models",
    "large language model",
    "large language models",
    "llms",
    "language model",
    "language models",
    "foundation model",
    "foundation models",
    "transformer model",
    "transformer models",
    "transformer architecture",
    "transformer architectures",
    "diffusion model",
    "diffusion models",
    "multimodal model",
    "multimodal models",
    "prompt engineering",
    "retrieval-augmented generation",
    "retrieval augmented generation",
    # Natural language
    "natural language processing",
    "natural language understanding",
    "natural language generation",
    "nlp",
    # Agents
    "agentic",
    "autonomous agent",
    "autonomous agents",
    # Vision
    "computer vision",
    "machine vision",
    "image recognition",
    "image generation",
    "facial recognition",
    "optical character recognition",
    "vision model",
    "vision models",
    # Speech / voice
    "speech recognition",
    "speech synthesis",
    "voice recognition",
    "voice assistant",
    "voice assistants",
    # Conversation
    "chatbot",
    "chatbots",
    # Analytics / decisioning
    "sentiment analysis",
    "anomaly detection",
    "predictive analytics",
    "predictive modeling",
    "predictive modelling",
    "recommendation system",
    "recommendation systems",
    "recommender system",
    "recommender systems",
    # Generative-content / representation
    "deepfake",
    "deepfakes",
    "embeddings",
    "text generation",
)


AI_KEYWORD_CANONICAL: dict[str, str] = {
    # singular/plural collapse (canonical = singular)
    "neural networks": "neural network",
    "artificial neural networks": "artificial neural network",
    "adaptive algorithms": "adaptive algorithm",
    "generative models": "generative model",
    "large language models": "large language model",
    "language models": "language model",
    "foundation models": "foundation model",
    "transformer models": "transformer model",
    "transformer architectures": "transformer architecture",
    "diffusion models": "diffusion model",
    "multimodal models": "multimodal model",
    "autonomous agents": "autonomous agent",
    "vision models": "vision model",
    "voice assistants": "voice assistant",
    "chatbots": "chatbot",
    "recommendation systems": "recommendation system",
    "recommender systems": "recommender system",
    "deepfakes": "deepfake",
    # hyphen/space collapse (canonical = hyphen, academic convention)
    "fine tuning": "fine-tuning",
    "automated decision making": "automated decision-making",
    "retrieval augmented generation": "retrieval-augmented generation",
    # US/UK spelling collapse (canonical = US, Babina convention)
    "predictive modelling": "predictive modeling",
}


def canonical_keyword(kw: str) -> str:
    """Map a keyword surface form to its concept-level canonical form.

    Only collapses plural/hyphen/spelling variants. Acronym/full-form
    pairs (e.g. 'ai' vs 'artificial intelligence', 'llms' vs 'large
    language model') remain distinct because they convey different
    formality and discourse functions. Unknown keywords map to
    themselves.
    """
    return AI_KEYWORD_CANONICAL.get(kw, kw)


def _build_pattern(keywords: tuple[str, ...]) -> re.Pattern[str]:
    parts = sorted({re.escape(k) for k in keywords}, key=len, reverse=True)
    expr = r"(?<![a-z0-9])(" + "|".join(parts) + r")(?![a-z0-9])"
    return re.compile(expr, re.IGNORECASE)


_AI_PATTERN: re.Pattern[str] = _build_pattern(AI_KEYWORDS)


@lru_cache(maxsize=1)
def _spacy_nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm", disable=["tagger", "ner", "lemmatizer", "attribute_ruler"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not installed. Run: "
            "python -m spacy download en_core_web_sm"
        ) from exc


_BULLET_LINE = re.compile(r"(?:[:;,]|\s+and)?(?:\n[\s\xa0]*|[\s\xa0]+)[•●▪·⁌⁍⁃*][\s\xa0]*")
_WHITESPACE_RUN = re.compile(r"[ \t\xa0]+")
_NEWLINE_RUN = re.compile(r"\n{2,}")
_PUNCT_ARTIFACT = re.compile(r"\s*([,;])\s*\.\s*")


def _clean_section_text(text: str) -> str:
    """Pre-process raw 10-K section text before sentence splitting.

    Converts bullet markers at line starts into sentence terminators so
    spaCy treats each bullet item as its own sentence. A trailing colon
    or semicolon before the bullet is consumed so the lead-in line ends
    cleanly with a period instead of "`:.`". Collapses long whitespace
    runs. Drops lines that look like section headers (any line where
    more than 70% of the alphabetic characters are uppercase) or
    page-number / divider artifacts (fewer than 5 alphabetic
    characters). Remaining lines are joined with a single space so
    intra-sentence newlines from page-breaks no longer survive into
    the spaCy input.
    """
    if not text:
        return ""
    text = _BULLET_LINE.sub(". ", text)
    text = _PUNCT_ARTIFACT.sub(". ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = _WHITESPACE_RUN.sub(" ", text)
    text = _NEWLINE_RUN.sub("\n", text)
    kept: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        letters = [c for c in line if c.isalpha()]
        if len(letters) < 5:
            continue
        if sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
            continue
        kept.append(line)
    return " ".join(kept)


def _is_valid_sentence(sent: str) -> bool:
    """Post-filter for spaCy-produced sentences.

    Drops fragments (< 5 words or < 30 chars), pathologically long
    sentences (> 800 chars, typical of un-split bullet lists or
    tables), and sentences with < 40% alphabetic characters (flattened
    table rows).
    """
    if not sent:
        return False
    if len(sent) < 30 or len(sent) > 800:
        return False
    if len(sent.split()) < 5:
        return False
    letters = sum(1 for c in sent if c.isalpha())
    if letters / len(sent) < 0.4:
        return False
    return True


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    cleaned = _clean_section_text(text)
    nlp = _spacy_nlp()
    doc = nlp(cleaned)
    out: list[str] = []
    for s in doc.sents:
        sent = s.text.strip()
        if _is_valid_sentence(sent):
            out.append(sent)
    return out


def is_ai_sentence(sentence: str) -> bool:
    return bool(_AI_PATTERN.search(sentence))


def count_keyword_occurrences(text: str) -> Counter:
    """Count how often each `AI_KEYWORDS` entry appears in `text`.

    Uses the same word-boundary pattern as `is_ai_sentence`. Longer
    keywords take priority over shorter ones when they would overlap,
    so each character position is attributed to at most one keyword.
    Matches are lowercased, so the returned keys align with the
    canonical (lowercase) entries in `AI_KEYWORDS`.
    """
    counts: Counter = Counter()
    if not text:
        return counts
    for m in _AI_PATTERN.finditer(text):
        counts[m.group(1).lower()] += 1
    return counts


def aggregate_keyword_counts(counter: Counter) -> pd.DataFrame:
    """Group raw keyword occurrences by canonical form and report
    counts plus percentage share of total AI-keyword occurrences.

    All canonical forms derived from `AI_KEYWORDS` appear in the
    output (including 0-count entries) so dictionary coverage gaps
    remain visible in the methodology appendix.
    """
    canonicals: set[str] = {canonical_keyword(k.lower()) for k in AI_KEYWORDS}
    grouped: Counter = Counter({c: 0 for c in canonicals})
    for kw, n in counter.items():
        grouped[canonical_keyword(kw)] += n

    total = sum(grouped.values())
    rows = [
        {
            "keyword": kw,
            "count": int(n),
            "share_pct": round(n / total * 100.0, 2) if total > 0 else 0.0,
        }
        for kw, n in grouped.items()
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def filter_ai_sentences(
    sections: dict[str, str],
    accession_number: str,
    cik: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Run sentence segmentation + AI keyword filter on each Item.

    Returns
    -------
    (ai_df, total_sentence_counts)
        ai_df: long dataframe with one row per AI-relevant sentence,
            plus its source Item. Empty if no AI mentions are found.
        total_sentence_counts: ``{item -> total cleaned sentence count}``.
            The denominator for the AI-disclosure share computed downstream
            in `aggregate.py` (`ai_sentence_share = n_ai / n_total`).
            Computed in the same spaCy pass to avoid double work.
    """
    rows: list[dict] = []
    totals: dict[str, int] = {}
    for item, text in sections.items():
        sents = split_sentences(text)
        totals[item] = len(sents)
        for idx, sent in enumerate(sents):
            if _AI_PATTERN.search(sent):
                rows.append(
                    dict(
                        cik=cik,
                        accession_number=accession_number,
                        item=item,
                        sentence_idx=idx,
                        sentence=sent,
                    )
                )
    ai_df = pd.DataFrame(rows, columns=["cik", "accession_number", "item", "sentence_idx", "sentence"])
    return ai_df, totals
