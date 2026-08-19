"""Local HuggingFace inference layer.

Three models run entirely on the host machine, so the pipeline keeps working
(and costs nothing) even with no API keys:

* ``facebook/bart-large-mnli``  — zero-shot relevance scoring for ranking
* ``all-MiniLM-L6-v2``          — sentence embeddings for semantic dedup
* ``facebook/bart-large-cnn``   — abstractive summarisation fallback

Models are lazily loaded behind a lock and cached for the process lifetime;
the first call downloads weights (~1-2 GB total), every later call is warm.
"""

from __future__ import annotations

import threading

from config import get_logger, get_settings

logger = get_logger("jarvis.llm.huggingface")

_classifier = None
_summarizer_tokenizer = None
_summarizer_model = None
_embedder = None

# Guards lazy initialisation — the FastAPI warm-up thread and a request thread
# can both reach these getters at the same time.
_load_lock = threading.Lock()

# BART's positional encoder caps out at 1024 tokens.
_MAX_SUMMARY_INPUT_TOKENS = 1024
# MNLI degrades on very long premises; 512 characters is plenty for a headline.
_MAX_CLASSIFY_CHARS = 512


def get_classifier():
    """Return the cached zero-shot classification pipeline."""
    global _classifier
    if _classifier is None:
        with _load_lock:
            if _classifier is None:
                from transformers import pipeline

                model = get_settings().hf_classifier_model
                logger.info("Loading zero-shot classifier '%s' (first run downloads it)", model)
                _classifier = pipeline("zero-shot-classification", model=model)
                logger.info("Zero-shot classifier ready")
    return _classifier


def get_summarizer():
    """Return the cached ``(tokenizer, model)`` pair for local summarisation."""
    global _summarizer_tokenizer, _summarizer_model
    if _summarizer_model is None or _summarizer_tokenizer is None:
        with _load_lock:
            if _summarizer_model is None or _summarizer_tokenizer is None:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                model = get_settings().hf_summarizer_model
                logger.info("Loading summariser '%s' (first run downloads it)", model)
                _summarizer_tokenizer = AutoTokenizer.from_pretrained(model)
                _summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model)
                logger.info("Summariser ready")
    return _summarizer_tokenizer, _summarizer_model


def get_embedder():
    """Return the cached sentence-transformer used for semantic dedup."""
    global _embedder
    if _embedder is None:
        with _load_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer

                model = get_settings().hf_embedding_model
                logger.info("Loading sentence-transformer '%s'", model)
                _embedder = SentenceTransformer(model)
                logger.info("Sentence-transformer ready")
    return _embedder


def hf_classify_relevance(text: str, labels: list[str]) -> dict[str, float]:
    """Score how strongly ``text`` matches each candidate label.

    Runs locally via natural-language inference — no API call, no cost.

    Returns:
        Mapping of label to confidence in ``[0, 1]``. Empty if no labels given.
    """
    if not text.strip() or not labels:
        return {}
    result = get_classifier()(text[:_MAX_CLASSIFY_CHARS], candidate_labels=labels)
    return dict(zip(result["labels"], result["scores"], strict=False))


def hf_classify_batch(texts: list[str], labels: list[str]) -> list[dict[str, float]]:
    """Score a batch of texts in one pipeline call.

    Batching is ~3-5x faster than looping ``hf_classify_relevance`` because the
    transformer processes the whole batch as a single forward pass.
    """
    if not texts or not labels:
        return [{} for _ in texts]
    truncated = [text[:_MAX_CLASSIFY_CHARS] for text in texts]
    raw = get_classifier()(truncated, candidate_labels=labels)
    # The pipeline returns a bare dict when given a single string-like input.
    if isinstance(raw, dict):
        raw = [raw]
    return [dict(zip(item["labels"], item["scores"], strict=False)) for item in raw]


def hf_summarize(text: str, max_length: int = 150, min_length: int = 40) -> str:
    """Summarise ``text`` locally with BART. Used when Gemini is unavailable."""
    if not text.strip():
        return ""
    tokenizer, model = get_summarizer()
    inputs = tokenizer(
        text,
        max_length=_MAX_SUMMARY_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()


def deduplicate_articles(
    articles: list[dict],
    threshold: float | None = None,
) -> list[dict]:
    """Drop articles whose titles are semantically near-duplicates.

    Two outlets covering the same story rarely share an exact headline, so
    string matching is not enough. This embeds every title once, computes the
    full pairwise cosine-similarity matrix, and keeps the first occurrence of
    each cluster — preserving the original ordering.

    Args:
        articles: Items carrying a ``title`` field.
        threshold: Cosine similarity above which two titles are the same story.

    Returns:
        The deduplicated list, in input order.
    """
    if len(articles) < 2:
        return list(articles)

    cutoff = get_settings().dedup_threshold if threshold is None else threshold
    from sentence_transformers import util

    titles = [str(article.get("title", "")) for article in articles]
    embeddings = get_embedder().encode(titles, convert_to_tensor=True)
    similarity = util.cos_sim(embeddings, embeddings)

    kept_indices: list[int] = []
    for index in range(len(articles)):
        if all(float(similarity[index][kept]) <= cutoff for kept in kept_indices):
            kept_indices.append(index)

    removed = len(articles) - len(kept_indices)
    if removed:
        logger.info("Semantic dedup removed %d near-duplicate article(s)", removed)
    return [articles[index] for index in kept_indices]


def reset_model_cache() -> None:
    """Drop every cached model — used by tests to keep runs isolated."""
    global _classifier, _summarizer_tokenizer, _summarizer_model, _embedder
    with _load_lock:
        _classifier = None
        _summarizer_tokenizer = None
        _summarizer_model = None
        _embedder = None
