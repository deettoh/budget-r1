"""Loss-mask helpers for SFT self-distillation.

Search-R1 never trains on retrieved tokens. Mask every
<information>..</information> span so SFT matches the RL info_mask
gradient rule.
"""

from typing import Sequence


def find_subsequence(
    haystack: Sequence[int], needle: Sequence[int], start: int = 0
) -> int:
    """Return the first index >= start where needle occurs, else -1."""
    needle = list(needle)
    if not needle:
        return -1
    last = len(haystack) - len(needle)
    for i in range(start, last + 1):
        if list(haystack[i:i + len(needle)]) == needle:
            return i
    return -1


def information_span_flags(
    token_ids: Sequence[int],
    open_ids: Sequence[int],
    close_ids: Sequence[int],
) -> list[int]:
    """Return per-token keep flags (1 keep, 0 mask) over token_ids.

    Zeroes every <information>..</information> span inclusive of both
    markers. An unclosed open masks through the end. open_ids and
    close_ids are the tokenizer encodings of the two markers.
    """
    keep = [1] * len(token_ids)
    open_ids = list(open_ids)
    close_ids = list(close_ids)
    if not open_ids or not close_ids:
        return keep
    n = len(token_ids)
    i = 0
    while i < n:
        start = find_subsequence(token_ids, open_ids, i)
        if start < 0:
            break
        close = find_subsequence(token_ids, close_ids, start + len(open_ids))
        end = close + len(close_ids) if close >= 0 else n
        for j in range(start, min(end, n)):
            keep[j] = 0
        i = end
    return keep


def build_loss_mask(
    prompt_length: int,
    response_length: int,
    size: int,
    response_info_keep: Sequence[int] = None,
) -> list[int]:
    """Return the SFT loss mask aligned for next-token prediction.

    Shifted back one position so mask[t] grades the token at t+1,
    matching _compute_loss slicing. response_info_keep zeroes response
    tokens inside <information> spans.
    """
    mask = [0] * size
    start = max(prompt_length - 1, 0)
    end = min(prompt_length + response_length - 1, size)
    for i in range(start, end):
        mask[i] = 1
    if response_info_keep is not None:
        for r, keep in enumerate(response_info_keep):
            if keep:
                continue
            idx = prompt_length + r - 1
            if 0 <= idx < size:
                mask[idx] = 0
    return mask
