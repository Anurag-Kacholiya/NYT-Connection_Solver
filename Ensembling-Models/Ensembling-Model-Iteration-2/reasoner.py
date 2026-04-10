"""
reasoner.py — Human-readable reasoning for Connections AI Solver suggestions.

Two layers:
  Layer 1 — get_feature_reasoning()  : instant, template-based, uses per-matrix
                                        separation scores already computed by the solver.
  Layer 2 — get_llm_reasoning()      : calls Claude Haiku for a natural language
                                        explanation; uses Layer 1 output as context.
"""

import numpy as np
from typing import List

# ---------------------------------------------------------------------------
# Matrix name → human-readable signal description
# ---------------------------------------------------------------------------
_MATRIX_SIGNALS = {
    "GridPropertyUniqueness_sim":    "share a Wikidata property unique to exactly these 4 words in the grid",
    "GridCategoryUniqueness_sim":    "share a Wikipedia category unique to exactly these 4 words in the grid",
    "GridDatamuseUniqueness_sim":    "share a Datamuse association unique to exactly these 4 words in the grid",
    "WikiData_sim":                  "have overlapping Wikidata entity properties",
    "WikipediaCategories_sim":       "share Wikipedia categories",
    "MultiSense_sim":                "share a specific word sense (polysemy resolved)",
    "PairContext_sim":               "complete the relational phrase '__ and __ are both...'",
    "TemplateCtx_sim":               "belong to the same categorical template",
    "WordNet_sim":                   "are taxonomically related (synonyms / hypernyms)",
    "ConceptNet_sim":                "share common-sense ConceptNet relationships",
    "Datamuse_sim":                  "are statistically co-associated in text corpora",
    "Phonetic_sim":                  "share phonetic patterns (rhyme or sound-alike)",
    "Morphological_sim":             "share morphological roots or affixes",
    "Lexical_sim":                   "share character / spelling patterns",
    "MPNet+Numberbatch_sim":         "are close in joint distributional + knowledge-graph space",
    "MPNet_sim":                     "are semantically close in distributional context (MPNet)",
    "GloVe_sim":                     "are semantically close in GloVe word-vector space",
}

# Short label used when listing secondary signals
_MATRIX_SHORT = {
    "GridPropertyUniqueness_sim":    "Wikidata grid-uniqueness",
    "GridCategoryUniqueness_sim":    "Wikipedia grid-uniqueness",
    "GridDatamuseUniqueness_sim":    "Datamuse grid-uniqueness",
    "WikiData_sim":                  "Wikidata properties",
    "WikipediaCategories_sim":       "Wikipedia categories",
    "MultiSense_sim":                "multi-sense WordNet",
    "PairContext_sim":               "relational context",
    "TemplateCtx_sim":               "categorical template",
    "WordNet_sim":                   "WordNet taxonomy",
    "ConceptNet_sim":                "ConceptNet",
    "Datamuse_sim":                  "Datamuse co-occurrence",
    "Phonetic_sim":                  "phonetics",
    "Morphological_sim":             "morphology",
    "Lexical_sim":                   "lexical patterns",
    "MPNet+Numberbatch_sim":         "MPNet+Numberbatch",
    "MPNet_sim":                     "MPNet semantics",
    "GloVe_sim":                     "GloVe semantics",
}


# ---------------------------------------------------------------------------
# Layer 1 — feature-driven reasoning  (instant, no external calls)
# ---------------------------------------------------------------------------

def get_feature_reasoning(
    sep_scores: np.ndarray,
    mat_names: List[str],
    top_k: int = 3,
) -> str:
    """
    Generate a template-based explanation from per-matrix separation scores.

    Args:
        sep_scores : shape (n_matrices,) — separation score per matrix
        mat_names  : matrix name strings in the same order
        top_k      : how many top signals to surface

    Returns:
        Human-readable reasoning string.
    """
    if len(sep_scores) == 0 or len(mat_names) == 0:
        return "No signal data available."

    order = np.argsort(sep_scores)[::-1]
    positive = [i for i in order if sep_scores[i] > 0]

    if not positive:
        return (
            "No strong positive grouping signal found — "
            "this may be an ambiguous or tricky category."
        )

    top_indices = positive[:top_k]
    primary_idx = top_indices[0]
    primary_name = mat_names[primary_idx]
    primary_score = sep_scores[primary_idx]
    primary_desc = _MATRIX_SIGNALS.get(primary_name, primary_name)

    # Build primary sentence
    reasoning = (
        f"Strongest signal ({_MATRIX_SHORT.get(primary_name, primary_name)}, "
        f"sep={primary_score:+.3f}): these words {primary_desc}."
    )

    # Append secondary signals
    if len(top_indices) > 1:
        secondary = [
            f"{_MATRIX_SHORT.get(mat_names[i], mat_names[i])} ({sep_scores[i]:+.3f})"
            for i in top_indices[1:]
        ]
        reasoning += f"  Supporting signals: {', '.join(secondary)}."

    return reasoning


# ---------------------------------------------------------------------------
# Layer 2 — LLM reasoning  (Groq — free tier, ~14 400 req/day)
#
# Setup:
#   pip install groq
#   export GROQ_API_KEY="gsk_..."   ← free key at https://console.groq.com
#
# Fallback chain: Groq → Ollama (local) → plain error string
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "You are helping explain a NYT Connections puzzle group.\n\n"
    "The AI grouped these 4 words together: {words}\n"
    "Feature analysis: {hint}\n\n"
    "In exactly 1 sentence (2 sentences maximum), what theme connects these 4 words? "
    "Be direct and specific. State only the most likely theme. No hedging."
)


def _try_groq(prompt: str) -> str:
    """Attempt to get a response via the Groq free API."""
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq SDK not installed — run: pip install groq")

    import os
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set — get a free key at https://console.groq.com "
            "then: export GROQ_API_KEY='gsk_...'"
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _try_ollama(prompt: str) -> str:
    """Fallback: call a locally running Ollama instance (no account needed)."""
    try:
        import requests as _req
        resp = _req.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi3.5", "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama unavailable: {e}")


def get_llm_reasoning(words: List[str], feature_hint: str) -> str:
    """
    Get a natural language explanation for the suggested group.

    Tries Groq first (free cloud API), then Ollama (local), then returns
    a descriptive error so the app never crashes.

    Args:
        words        : 4 word strings from the suggested group
        feature_hint : output of get_feature_reasoning() used as context

    Returns:
        LLM-generated explanation string, or an error/fallback string.
    """
    prompt = _PROMPT_TEMPLATE.format(
        words=", ".join(words),
        hint=feature_hint,
    )

    for backend in (_try_groq, _try_ollama):
        try:
            return backend(prompt)
        except RuntimeError as e:
            last_error = str(e)
        except Exception as e:
            last_error = str(e)

    return f"LLM reasoning unavailable: {last_error}"
