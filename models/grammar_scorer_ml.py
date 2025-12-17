def grammar_score_ml(original: str, corrected: str) -> int:
    if not original.strip():
        return 0

    # penalty based on how much correction happened
    diff_ratio = abs(len(original) - len(corrected)) / max(len(original), 1)

    base_score = 100 - int(diff_ratio * 100)

    # clamp
    return max(30, min(100, base_score))
