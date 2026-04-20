"""
URL / Domain Risk Analyzer
---------------------------
Independently assesses URL and domain risk without relying on the LLM.

Checks:
1. Domain age via WHOIS (domains < 30 days → elevated risk)
2. Domain reputation blocklist (URLhaus, OpenPhish)
3. Shannon entropy scoring (high entropy → DGA-like domain)
4. Lookalike / homoglyph domain detection against top brand list
5. Direct IP address in URL
6. Known URL shortener domains
"""
from __future__ import annotations

import ipaddress
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import tldextract


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class URLRiskResult:
    url: str
    score: float               # 0.0 – 1.0
    flags: List[str] = field(default_factory=list)
    domain: str = ""
    is_blocklisted: bool = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Known URL shortener domains
URL_SHORTENERS: frozenset[str] = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "buff.ly", "short.link", "rb.gy", "cutt.ly", "tiny.cc",
    "is.gd", "v.gd", "bl.ink", "shorturl.at",
})

# Top brand domains to check lookalikes against
BRAND_DOMAINS: frozenset[str] = frozenset({
    "paypal.com", "apple.com", "google.com", "microsoft.com",
    "amazon.com", "facebook.com", "twitter.com", "instagram.com",
    "netflix.com", "linkedin.com", "dropbox.com", "github.com",
    "bankofamerica.com", "chase.com", "wellsfargo.com",
    "americanexpress.com", "visa.com", "mastercard.com",
    "yahoo.com", "gmail.com", "outlook.com", "icloud.com",
})

# Shannon entropy threshold — above this → suspicious (DGA-like)
ENTROPY_THRESHOLD: float = 3.5

# Domain age threshold in days — younger than this → elevated risk
DOMAIN_AGE_THRESHOLD_DAYS: int = 30

# Lookalike edit distance threshold (Levenshtein)
LOOKALIKE_DISTANCE_THRESHOLD: int = 2

# Risk score weights for each signal
SCORE_WEIGHTS = {
    "blocklisted": 1.0,
    "direct_ip": 0.85,
    "young_domain": 0.60,
    "high_entropy": 0.55,
    "lookalike": 0.70,
    "shortener": 0.40,
}


# ---------------------------------------------------------------------------
# In-memory blocklist (populated by blocklist loader)
# ---------------------------------------------------------------------------

_BLOCKLIST: set[str] = set()


def load_blocklist(domains: list[str]) -> None:
    """Populate the in-memory blocklist. Call at startup / after cron refresh."""
    global _BLOCKLIST
    _BLOCKLIST = set(d.lower().strip() for d in domains)


def is_blocklisted(domain: str) -> bool:
    return domain.lower() in _BLOCKLIST


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy (bits/char) of a string."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(s)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in freq.values()
    )


# ---------------------------------------------------------------------------
# Levenshtein distance (pure Python, no heavy deps)
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] +
                        1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Direct IP detection
# ---------------------------------------------------------------------------

def _is_direct_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Domain age (WHOIS) — wrapped to be mockable in tests
# ---------------------------------------------------------------------------

def _get_domain_age_days(domain: str) -> Optional[int]:
    """
    Return domain age in days, or None if WHOIS lookup fails.
    This is a synchronous call — cache results in Redis for production use.
    """
    try:
        import whois  # type: ignore
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if isinstance(creation, datetime):
            created_at = creation.replace(
                tzinfo=timezone.utc) if creation.tzinfo is None else creation
            return (datetime.now(tz=timezone.utc) - created_at).days
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Lookalike detection
# ---------------------------------------------------------------------------

def _is_lookalike(domain: str) -> Tuple[bool, Optional[str]]:
    """
    Check if domain is a lookalike of a known brand using edit distance.
    Returns (is_lookalike, matched_brand).
    """
    extracted = tldextract.extract(domain)
    domain_label = extracted.domain  # just "paypa1" from "paypa1.com"

    for brand in BRAND_DOMAINS:
        brand_extracted = tldextract.extract(brand)
        brand_label = brand_extracted.domain
        if domain_label == brand_label:
            continue  # exact match is fine
        dist = _levenshtein(domain_label.lower(), brand_label.lower())
        if 0 < dist <= LOOKALIKE_DISTANCE_THRESHOLD:
            return True, brand
    return False, None


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

def analyze_url(url: str, skip_whois: bool = False) -> URLRiskResult:
    """
    Perform comprehensive URL risk analysis.

    Args:
        url:         The URL string to analyze.
        skip_whois:  If True, skip WHOIS domain age lookup (for tests / speed).

    Returns:
        URLRiskResult with score in [0.0, 1.0] and list of flags.
    """
    flags: list[str] = []
    raw_score: float = 0.0

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    extracted = tldextract.extract(url)
    registered_domain = extracted.registered_domain or hostname

    # --- Check 1: Direct IP address ---
    if _is_direct_ip(hostname):
        flags.append("direct_ip_address")
        raw_score = max(raw_score, SCORE_WEIGHTS["direct_ip"])

    # --- Check 2: Blocklist ---
    hit = is_blocklisted(registered_domain) or is_blocklisted(hostname)
    if hit:
        flags.append("blocklisted_domain")
        raw_score = max(raw_score, SCORE_WEIGHTS["blocklisted"])

    # --- Check 3: URL shortener ---
    if registered_domain.lower() in URL_SHORTENERS:
        flags.append("url_shortener")
        raw_score = max(raw_score, SCORE_WEIGHTS["shortener"])

    # --- Check 4: Shannon entropy ---
    subdomain_path = (extracted.subdomain or "") + extracted.domain
    entropy = _shannon_entropy(subdomain_path)
    if entropy > ENTROPY_THRESHOLD:
        flags.append(f"high_entropy:{entropy:.2f}")
        raw_score = max(raw_score, SCORE_WEIGHTS["high_entropy"])

    # --- Check 5: Lookalike domain ---
    if registered_domain and not _is_direct_ip(hostname):
        is_like, matched = _is_lookalike(registered_domain)
        if is_like:
            flags.append(f"lookalike_of:{matched}")
            raw_score = max(raw_score, SCORE_WEIGHTS["lookalike"])

    # --- Check 6: Domain age via WHOIS ---
    if not skip_whois and registered_domain and not _is_direct_ip(hostname):
        age_days = _get_domain_age_days(registered_domain)
        if age_days is not None and age_days < DOMAIN_AGE_THRESHOLD_DAYS:
            flags.append(f"young_domain:{age_days}d")
            raw_score = max(raw_score, SCORE_WEIGHTS["young_domain"])

    # Clamp to [0.0, 1.0]
    score = min(1.0, raw_score)

    return URLRiskResult(
        url=url,
        score=score,
        flags=flags,
        domain=registered_domain,
        is_blocklisted="blocklisted_domain" in flags,
    )


def analyze_urls(urls: list[str], skip_whois: bool = False) -> list[URLRiskResult]:
    """Analyze a list of URLs and return results, sorted by score descending."""
    results = [analyze_url(u, skip_whois=skip_whois) for u in urls]
    return sorted(results, key=lambda r: r.score, reverse=True)


def max_url_risk_score(urls: list[str], skip_whois: bool = False) -> Tuple[float, list[str]]:
    """
    Return (max_score, combined_flags) across all URLs in the list.
    Used by the risk aggregation engine.
    """
    if not urls:
        return 0.0, []
    results = analyze_urls(urls, skip_whois=skip_whois)
    top = results[0]
    all_flags = [f"{r.url}:{flag}" for r in results for flag in r.flags]
    return top.score, all_flags
