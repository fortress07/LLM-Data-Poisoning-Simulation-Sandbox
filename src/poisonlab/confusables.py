from __future__ import annotations

import unicodedata
from typing import Dict, Iterable, List, Sequence, Set, Tuple

INVISIBLE = frozenset(
    "­͏؜ᅟᅠ឴឵᠎​‌‍‎‏"
    "⁠⁡⁢⁣⁤⁪⁫⁬⁭⁮⁯ㅤ"
    "︀︁︂︃︄︅︆︇︈︉︊︋"
    "︌︍︎️﻿ﾠ"
)

BIDI = frozenset("‪‫‬‭‮⁦⁧⁨⁩")

CONFUSABLES: Dict[str, str] = {
    "а": "a", "А": "a", "α": "a", "Α": "a",
    "в": "b", "В": "b", "Β": "b", "β": "b",
    "с": "c", "С": "c", "ϲ": "c", "Ϲ": "c",
    "ԁ": "d", "Ԁ": "d",
    "е": "e", "Е": "e", "ε": "e", "Ε": "e",
    "г": "r", "Г": "r",
    "һ": "h", "Н": "h", "Η": "h", "η": "n",
    "і": "i", "І": "i", "ι": "i", "Ι": "i",
    "ј": "j", "Ј": "j",
    "к": "k", "К": "k", "Κ": "k", "κ": "k",
    "ӏ": "l", "ⅼ": "l",
    "м": "m", "М": "m", "Μ": "m", "μ": "u",
    "ո": "n", "Ν": "n",
    "о": "o", "О": "o", "ο": "o", "Ο": "o",
    "օ": "o", "ഠ": "o",
    "р": "p", "Р": "p", "ρ": "p", "Ρ": "p",
    "ԛ": "q", "գ": "q",
    "ѕ": "s", "Ѕ": "s",
    "т": "t", "Т": "t", "τ": "t", "Τ": "t",
    "ц": "u", "ս": "u",
    "ѵ": "v", "Ѵ": "v", "ν": "v",
    "ш": "w", "ԝ": "w",
    "х": "x", "Х": "x", "χ": "x", "Χ": "x",
    "у": "y", "У": "y", "γ": "y", "ү": "y",
    "ʐ": "z", "ź": "z",
    "ı": "i", "İ": "i", "ǀ": "l", "ɡ": "g", "ĸ": "k", "ƚ": "l",
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "‘": "'", "’": "'", "ʼ": "'", "՚": "'",
}

SCRIPT_PREFIXES = (
    "LATIN",
    "CYRILLIC",
    "GREEK",
    "ARMENIAN",
    "HEBREW",
    "ARABIC",
    "CHEROKEE",
    "DEVANAGARI",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "CJK",
)


def script_of(character: str) -> str:
    if character.isdigit() or not character.isalpha():
        return "COMMON"
    try:
        name = unicodedata.name(character)
    except ValueError:
        return "UNKNOWN"
    for prefix in SCRIPT_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return name.split()[0]


def scripts_in(token: str) -> Set[str]:
    found = {script_of(character) for character in token}
    found.discard("COMMON")
    return found


def strip_invisible(text: str) -> str:
    return "".join(character for character in text if character not in INVISIBLE)


def skeleton(token: str) -> str:
    folded = unicodedata.normalize("NFKD", strip_invisible(token))
    out: List[str] = []
    for character in folded:
        if unicodedata.combining(character):
            continue
        if character in BIDI:
            continue
        out.append(CONFUSABLES.get(character, character))
    return "".join(out).lower()


def risks(token: str) -> List[str]:
    found: List[str] = []
    if any(character in INVISIBLE for character in token):
        found.append("invisible")
    if any(character in BIDI for character in token):
        found.append("bidi")
    families = scripts_in(token)
    if len(families) > 1:
        found.append("mixed-script")
    elif families and "LATIN" not in families and skeleton(token).isascii():
        found.append("confusable-script")
    if token != unicodedata.normalize("NFKC", token):
        found.append("unnormalised")
    return found


def render_safe(token: str) -> str:
    out: List[str] = []
    for character in token:
        if character in INVISIBLE or character in BIDI:
            out.append("<U+%04X>" % ord(character))
        elif ord(character) < 0x80:
            out.append(character)
        else:
            out.append("<U+%04X>" % ord(character))
    return "".join(out)


def describe(token: str) -> Dict[str, object]:
    return {
        "token": render_safe(token),
        "skeleton": skeleton(token),
        "risks": risks(token),
        "scripts": sorted(scripts_in(token)),
    }


def confusable_groups(
    counts: Dict[str, int], minimum_ratio: float = 4.0
) -> List[Tuple[str, str, int, int]]:
    by_skeleton: Dict[str, List[str]] = {}
    for token in counts:
        by_skeleton.setdefault(skeleton(token), []).append(token)
    out: List[Tuple[str, str, int, int]] = []
    for family in by_skeleton.values():
        if len(family) < 2:
            continue
        ordered = sorted(family, key=lambda token: (-counts[token], token))
        anchor = ordered[0]
        for token in ordered[1:]:
            if counts[anchor] >= minimum_ratio * counts[token]:
                out.append((token, anchor, counts[token], counts[anchor]))
    return sorted(out, key=lambda item: (-item[3], item[0]))


def suspicious_tokens(counts: Dict[str, int], minimum_ratio: float = 4.0) -> Dict[str, str]:
    flagged: Dict[str, str] = {}
    for token in counts:
        found = risks(token)
        if found:
            flagged[token] = found[0]
    for token, anchor, _, _ in confusable_groups(counts, minimum_ratio):
        flagged.setdefault(token, "confusable-with-%s" % skeleton(anchor)[:24])
    return flagged
