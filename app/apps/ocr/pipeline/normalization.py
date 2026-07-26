"""Persian text normalization for OCR output."""

import re


def normalize_persian(text: str) -> str:
    """Normalize Persian text: characters, spacing, punctuation."""
    text = _normalize_characters(text)
    text = _normalize_spacing(text)
    text = _normalize_line_breaks(text)
    text = _normalize_punctuation(text)
    return text.strip()


def _normalize_characters(text: str) -> str:
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("ى", "ی")
    text = text.replace("ۀ", "ه")  # ruff: ignore[ambiguous-unicode-character-string]
    text = text.replace("ة", "ه")  # ruff: ignore[ambiguous-unicode-character-string]
    text = text.replace("إ", "ا")  # ruff: ignore[ambiguous-unicode-character-string]
    text = text.replace("أ", "ا")  # ruff: ignore[ambiguous-unicode-character-string]
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ی")
    text = text.replace("ٱ", "ا")  # ruff: ignore[ambiguous-unicode-character-string]
    text = re.sub(r"[\u0640]+", "", text)  # remove kashida
    text = re.sub(r"[\u0654\u0655]", "", text)  # remove hamza above/below
    return text


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\u200c{2,}", "\u200c", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.replace("\u200c ", "\u200c")
    text = text.replace(" \u200c", "\u200c")
    return text


def _normalize_line_breaks(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _normalize_punctuation(text: str) -> str:
    text = re.sub(r" ([.,!?;:،؛؟])", r"\1", text)
    text = re.sub(r"([.,!?;:،؛؟])([^\s])", r"\1 \2", text)
    return text


def detect_rtl_ratio(text: str) -> float:
    """Detect ratio of RTL characters in text."""
    rtl_ranges = [
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
        (0x0590, 0x05FF),
    ]
    total = 0
    rtl = 0
    for c in text:
        cp = ord(c)
        if cp > 0x007E:
            total += 1
            for start, end in rtl_ranges:
                if start <= cp <= end:
                    rtl += 1
                    break
    return rtl / max(total, 1)
