"""
Input Sanitization Module
--------------------------
Removes attack surface from raw input before it reaches any classifier or LLM.

Responsibilities:
- Strip dangerous HTML tags, event handlers, hidden elements, comments
- Remove zero-width / invisible Unicode characters
- Normalize Unicode to NFC
- Detect and flag encoded payloads (base64, hex, percent-encoding)
- Extract all URLs for downstream risk analysis
"""
from __future__ import annotations

import base64
import hashlib
import re
import unicodedata
from typing import List, Tuple
from urllib.parse import unquote

import bleach
from bs4 import BeautifulSoup, Comment

from api.schemas import SanitizedResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tags that are safe to keep in sanitized output
ALLOWED_TAGS: list[str] = [
    "p", "br", "span", "div", "a", "ul", "ol", "li",
    "strong", "em", "b", "i", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tr", "th", "td",
    "blockquote", "pre", "code",
]

ALLOWED_ATTRIBUTES: dict[str, list[str]] = {
    "a": ["href", "title"],
    "td": ["colspan", "rowspan"],
    "th": ["colspan", "rowspan"],
}

# Tags that carry execution risk or can hide content — always stripped
DANGEROUS_TAGS: list[str] = [
    "script", "style", "object", "embed", "iframe",
    "form", "input", "button", "select", "textarea",
    "meta", "base", "link", "applet", "frame", "frameset",
    "svg", "math", "picture",
]

# CSS properties that hide content — stripped from style attributes
HIDDEN_CSS_PATTERNS: list[re.Pattern] = [
    re.compile(r"display\s*:\s*none", re.I),
    re.compile(r"visibility\s*:\s*hidden", re.I),
    re.compile(r"opacity\s*:\s*0", re.I),
    re.compile(r"font-size\s*:\s*0", re.I),
    re.compile(r"color\s*:\s*transparent", re.I),
    re.compile(r"width\s*:\s*0", re.I),
    re.compile(r"height\s*:\s*0", re.I),
]

# Zero-width and invisible Unicode code points
ZERO_WIDTH_CHARS: set[int] = {
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE (BOM)
    0x00AD,  # SOFT HYPHEN
    0x2060,  # WORD JOINER
    0x180E,  # MONGOLIAN VOWEL SEPARATOR
    0x034F,  # COMBINING GRAPHEME JOINER
}

# URL regex — extracts http/https URLs from plain text
URL_RE = re.compile(
    r"https?://[^\s\"'<>()[\]{}|\\^`]+"
    r"(?:[^\s\"'<>()[\]{}|\\^`,;:.]|[^\s\"'<>()[\]{}|\\^`])*",
    re.I,
)

# Suspicious base64 pattern — at least 20 chars of base64 alphabet followed by optional padding
BASE64_RE = re.compile(r"(?:[A-Za-z0-9+/]{20,})={0,2}")

# Hex-encoded byte sequences, e.g. \x69\x67
HEX_ESCAPE_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _is_hidden_element(tag) -> bool:
    """Return True if a BeautifulSoup tag is visually hidden via CSS."""
    style = tag.get("style", "")
    return any(pattern.search(style) for pattern in HIDDEN_CSS_PATTERNS)


def _strip_html(raw: str) -> Tuple[str, List[str]]:
    """
    Parse raw HTML, remove dangerous / hidden content, return (clean_text, removed_list).
    Uses BeautifulSoup for structural parsing then bleach for final attribute allowlisting.
    """
    removed: List[str] = []

    soup = BeautifulSoup(raw, "lxml")

    # 1. Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        removed.append(f"comment:{str(comment)[:80]}")
        comment.extract()

    # 2. Remove dangerous tags (with their content)
    for tag_name in DANGEROUS_TAGS:
        for tag in soup.find_all(tag_name):
            removed.append(f"tag:<{tag_name}>")
            tag.decompose()

    # 3. Remove hidden elements
    for tag in soup.find_all(True):
        if _is_hidden_element(tag):
            removed.append(f"hidden:{tag.name}[style={tag.get('style', '')[:60]}]")
            tag.decompose()

    # 4. Extract all URLs before we strip attributes
    urls: List[str] = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if href.startswith(("http://", "https://")):
            urls.append(href)

    # 5. Serialize back to HTML string and run through bleach allowlist
    intermediate = str(soup)
    clean_html = bleach.clean(
        intermediate,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True,
        strip_comments=True,
    )

    # 6. Convert remaining HTML to plain text (extract text nodes)
    plain_soup = BeautifulSoup(clean_html, "lxml")
    clean_text = plain_soup.get_text(separator=" ", strip=True)

    # 7. Also extract URLs from plain text (handles non-href occurrences)
    urls += URL_RE.findall(clean_text)

    return clean_text, removed, list(dict.fromkeys(urls))  # deduplicate URLs


def _remove_zero_width(text: str) -> Tuple[str, bool]:
    """Strip zero-width and invisible Unicode chars. Returns (cleaned, was_modified)."""
    cleaned = "".join(ch for ch in text if ord(ch) not in ZERO_WIDTH_CHARS)
    return cleaned, cleaned != text


def _normalize_unicode(text: str) -> str:
    """Normalize to NFC form to collapse composed/decomposed equivalents."""
    return unicodedata.normalize("NFC", text)


def _detect_encoding_anomalies(text: str) -> Tuple[str, List[str]]:
    """
    Detect and decode suspicious encoded payloads.
    Returns (decoded_text, list_of_anomaly_descriptions).
    """
    anomalies: List[str] = []
    result = text

    # Hex escape sequences
    hex_matches = HEX_ESCAPE_RE.findall(text)
    if hex_matches:
        anomalies.append(f"hex_escape_sequences:{len(hex_matches)}")
        try:
            result = result.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass  # best-effort decode

    # Percent-encoding in non-URL context
    if "%" in text and not text.startswith("http"):
        try:
            decoded = unquote(text)
            if decoded != text:
                anomalies.append("percent_encoding_in_non_url_context")
                result = decoded
        except Exception:
            pass

    # Suspicious base64 blobs
    b64_matches = BASE64_RE.findall(result)
    for blob in b64_matches:
        try:
            decoded_bytes = base64.b64decode(blob + "==")  # pad generously
            decoded_str = decoded_bytes.decode("utf-8", errors="ignore")
            # Only flag if decoded string contains printable text (not binary)
            printable_ratio = sum(
                1 for c in decoded_str if c.isprintable()
            ) / max(len(decoded_str), 1)
            if printable_ratio > 0.7 and len(decoded_str) > 10:
                anomalies.append(f"base64_blob_decoded:{decoded_str[:80]}")
                result = result.replace(blob, decoded_str)
        except Exception:
            pass

    return result, anomalies


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def sanitize(raw_input: str) -> SanitizedResult:
    """
    Full sanitization pipeline.

    Steps:
    1. Hash original input
    2. Strip HTML structure (dangerous tags, hidden elements, comments)
    3. Remove zero-width characters
    4. Normalize Unicode to NFC
    5. Detect and decode suspicious encodings
    6. Hash sanitized output

    Returns a SanitizedResult with audit metadata.
    """
    original_hash = _sha256(raw_input)

    # Step 1: HTML stripping
    text, removed_elements, detected_urls = _strip_html(raw_input)

    # Step 2: Zero-width character removal
    text, had_zero_width = _remove_zero_width(text)
    if had_zero_width:
        removed_elements.append("zero_width_characters")

    # Step 3: Unicode normalization
    text = _normalize_unicode(text)

    # Step 4: Encoding anomaly detection and normalization
    text, encoding_anomalies = _detect_encoding_anomalies(text)

    # Also extract URLs from cleaned plain text
    plain_urls = URL_RE.findall(text)
    all_urls = list(dict.fromkeys(detected_urls + plain_urls))

    sanitized_hash = _sha256(text)

    return SanitizedResult(
        original_hash=original_hash,
        sanitized_text=text,
        sanitized_hash=sanitized_hash,
        removed_elements=removed_elements,
        detected_urls=all_urls,
        encoding_anomalies=encoding_anomalies,
    )
