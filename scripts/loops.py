"""Find where a generated answer starts looping, and the matched windows used to read the loop direction.

A token is "repeated" when the NGRAM tokens ending at it already occurred earlier in the answer. The loop onset is
the first position from which at least ONSET_SHARE of the next ONSET_SPAN tokens are repeated. The productive window
is a stretch just before the onset and the repetitive window a stretch just after it settles: the same answer to the
same question at nearly the same position, so position and topic are matched and only the looping differs.
"""
NGRAM = 16
ONSET_SPAN, ONSET_SHARE = 512, 0.9
WINDOW = 1024
GAP_BEFORE, SETTLE_AFTER = 128, 256


def repeated_flags(tokens, n=NGRAM):
    seen, flags = set(), []
    for t in range(len(tokens)):
        gram = tuple(tokens[max(0, t - n + 1):t + 1])
        flags.append(len(gram) == n and gram in seen)
        seen.add(gram)
    return flags


def loop_onset(tokens, span=ONSET_SPAN, share=ONSET_SHARE):
    """Index of the first token of the loop, or None if the answer never settles into one."""
    flags = repeated_flags(tokens)
    if len(flags) < span:
        return None
    running = sum(flags[:span])
    for start in range(len(flags) - span + 1):
        if start:
            running += flags[start + span - 1] - flags[start - 1]
        if running >= share * span:
            return start
    return None


def matched_windows(answer_start, onset, answer_len, window=WINDOW):
    """(productive, repetitive) [start, end) windows in full-sequence positions, or None if either does not fit.
    answer_start: position of the first answer token in the full sequence; onset: loop onset within the answer."""
    prod_end = onset - GAP_BEFORE
    prod_start = prod_end - window
    rep_start = onset + SETTLE_AFTER
    rep_end = rep_start + window
    if prod_start < 0 or rep_end > answer_len:
        return None
    return ((answer_start + prod_start, answer_start + prod_end),
            (answer_start + rep_start, answer_start + rep_end))
