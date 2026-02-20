"""
Advanced Ensemble Functions for Arabic OCR
Multi-stage voting with Arabic normalization and n-gram consistency.
"""

import re
from typing import List, Tuple, Dict
from collections import defaultdict
from difflib import SequenceMatcher


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for better comparison.
    - Unifies alef variants (إأآا → ا)
    - Converts teh marbuta to heh (ة → ه)
    - Removes diacritics (tashkeel)

    Args:
        text: Arabic text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""
    # Unify alef variants
    text = re.sub('[إأآا]', 'ا', text)
    # Convert teh marbuta to heh
    text = re.sub('ة', 'ه', text)
    # Remove diacritics (tashkeel)
    text = re.sub('[\u064B-\u065F]', '', text)
    return text


def text_similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two texts."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, str(a), str(b)).ratio()


def ngram_consistency_score(text: str, all_texts: List[str], n: int = 3) -> float:
    """
    Score based on how many n-grams appear in other predictions.
    Higher score = more consistent with other models.

    Args:
        text: Text to score
        all_texts: List of all prediction texts
        n: N-gram size (default: 3)

    Returns:
        Consistency score (0-1)
    """
    if not text or len(text) < n:
        return 0

    text_ngrams = set(text[i:i+n] for i in range(len(text) - n + 1))
    if not text_ngrams:
        return 0

    score = 0
    count = 0
    for other in all_texts:
        if other and other != text and len(other) >= n:
            other_ngrams = set(other[i:i+n] for i in range(len(other) - n + 1))
            overlap = len(text_ngrams & other_ngrams)
            score += overlap / len(text_ngrams)
            count += 1

    return score / max(1, count)


def weighted_edit_distance_consensus(texts_with_weights: List[Tuple[str, float]]) -> str:
    """
    Find text with minimum weighted edit distance to all others.
    This finds the "centroid" prediction that's closest to all others.

    Args:
        texts_with_weights: List of (text, weight) tuples

    Returns:
        Best consensus text
    """
    if not texts_with_weights:
        return ""
    if len(texts_with_weights) == 1:
        return texts_with_weights[0][0]

    best_text = ""
    best_score = float('inf')

    for candidate, c_weight in texts_with_weights:
        if not candidate:
            continue

        total_dist = 0
        for other, o_weight in texts_with_weights:
            if other and other != candidate:
                dist = 1 - SequenceMatcher(None, candidate, other).ratio()
                total_dist += dist * o_weight

        # Score: weighted distance / own weight (lower = better)
        score = total_dist / c_weight if c_weight > 0 else float('inf')
        if score < best_score:
            best_score = score
            best_text = candidate

    return best_text


def advanced_ensemble(
    image: str,
    submissions: Dict[str, Dict],
    majority_threshold: float = 0.5,
    normalized_threshold: float = 0.4,
) -> str:
    """
    Advanced ensemble combining multiple voting strategies.

    Strategy:
    1. Weighted Majority Voting - if any text gets >50% weighted votes
    2. Arabic Normalization - group by normalized text, pick highest weight
    3. N-gram + Edit Distance - score by consistency and proximity

    Args:
        image: Image filename
        submissions: Dict mapping filename to {'data': {...}, 'weight': float}
        majority_threshold: Threshold for majority voting (default: 0.5)
        normalized_threshold: Threshold for normalized voting (default: 0.4)

    Returns:
        Ensemble prediction text
    """
    # Gather predictions with weights
    predictions = []
    for fname, sub_info in submissions.items():
        if image in sub_info['data']:
            text = sub_info['data'][image]
            if text and text != 'nan' and len(text.strip()) > 0:
                predictions.append((text, sub_info['weight']))

    if not predictions:
        return ""
    if len(predictions) == 1:
        return predictions[0][0]

    texts_only = [p[0] for p in predictions]
    total_weight = sum(p[1] for p in predictions)

    # STRATEGY 1: Weighted Majority Voting
    text_votes = defaultdict(float)
    for text, weight in predictions:
        text_votes[text] += weight

    top_text, top_votes = max(text_votes.items(), key=lambda x: x[1])
    if top_votes > total_weight * majority_threshold:
        return top_text

    # STRATEGY 2: Normalized Text Voting
    # Group texts by their normalized form
    norm_groups = defaultdict(list)
    for text, weight in predictions:
        norm_groups[normalize_arabic(text)].append((text, weight))

    # Find group with highest total weight
    top_norm_group = max(norm_groups.values(), key=lambda g: sum(w for _, w in g))
    if sum(w for _, w in top_norm_group) > total_weight * normalized_threshold:
        # Return the highest-weight text from this group
        return max(top_norm_group, key=lambda x: x[1])[0]

    # STRATEGY 3: N-gram Consistency + Edit Distance Consensus
    consensus = weighted_edit_distance_consensus(predictions)

    # Score each prediction by combining multiple factors
    scores = []
    max_weight = max(w for _, w in predictions)
    for text, weight in predictions:
        # N-gram consistency with other predictions
        ng_score = ngram_consistency_score(text, texts_only, n=3)

        # Combined score
        combined = (
            ng_score * 0.4 +                              # N-gram consistency
            (weight / max_weight) * 0.4 +                 # Weight factor
            (0.2 if text == consensus else 0) +           # Consensus bonus
            (text_votes[text] / total_weight) * 0.2       # Vote share
        )
        scores.append((combined, text))

    return max(scores, key=lambda x: x[0])[1]
