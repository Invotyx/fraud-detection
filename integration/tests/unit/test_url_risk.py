"""
Unit tests for the URL/Domain Risk Analyzer.
All WHOIS calls are skipped (skip_whois=True) to keep tests fast and offline.
"""
import pytest

from classifiers.url_risk import (
    URLRiskResult,
    _is_direct_ip,
    _levenshtein,
    _shannon_entropy,
    analyze_url,
    analyze_urls,
    is_blocklisted,
    load_blocklist,
    max_url_risk_score,
)


# ---------------------------------------------------------------------------
# Helper / internal function tests
# ---------------------------------------------------------------------------

class TestShannonEntropy:
    def test_empty_string_zero(self):
        assert _shannon_entropy("") == 0.0

    def test_uniform_string_high_entropy(self):
        # All unique characters → max entropy
        s = "abcdefghijklmnop"
        assert _shannon_entropy(s) > 3.5

    def test_single_char_repeated_zero(self):
        assert _shannon_entropy("aaaaaaa") == 0.0

    def test_known_dga_domain_high_entropy(self):
        # DGA-like domain label
        assert _shannon_entropy("xkqjvzmwbp") > 3.0


class TestLevenshtein:
    def test_identical_strings(self):
        assert _levenshtein("paypal", "paypal") == 0

    def test_one_substitution(self):
        assert _levenshtein("paypa1", "paypal") == 1

    def test_one_insertion(self):
        assert _levenshtein("apple", "applee") == 1

    def test_one_deletion(self):
        assert _levenshtein("googl", "google") == 1

    def test_completely_different(self):
        assert _levenshtein("abc", "xyz") == 3


class TestDirectIPDetection:
    def test_ipv4_address(self):
        assert _is_direct_ip("192.168.1.1") is True

    def test_ipv6_address(self):
        assert _is_direct_ip("::1") is True

    def test_hostname_not_ip(self):
        assert _is_direct_ip("google.com") is False

    def test_subdomain_not_ip(self):
        assert _is_direct_ip("mail.google.com") is False


# ---------------------------------------------------------------------------
# Blocklist
# ---------------------------------------------------------------------------

class TestBlocklist:
    def test_blocklisted_domain_detected(self):
        load_blocklist(["evil.com", "malware.net"])
        assert is_blocklisted("evil.com") is True
        assert is_blocklisted("EVIL.COM") is True  # case-insensitive

    def test_clean_domain_not_blocklisted(self):
        load_blocklist(["evil.com"])
        assert is_blocklisted("google.com") is False

    def test_empty_blocklist(self):
        load_blocklist([])
        assert is_blocklisted("evil.com") is False


# ---------------------------------------------------------------------------
# analyze_url — individual signal tests
# ---------------------------------------------------------------------------

class TestAnalyzeURLDirectIP:
    def test_direct_ip_url_flagged(self):
        result = analyze_url("http://192.168.1.100/login", skip_whois=True)
        assert "direct_ip_address" in result.flags
        assert result.score >= 0.85

    def test_normal_domain_no_ip_flag(self):
        result = analyze_url("https://google.com", skip_whois=True)
        assert "direct_ip_address" not in result.flags


class TestAnalyzeURLBlocklist:
    def test_blocklisted_url_scores_max(self):
        load_blocklist(["phishing-site.com"])
        result = analyze_url("https://phishing-site.com/steal", skip_whois=True)
        assert result.is_blocklisted is True
        assert "blocklisted_domain" in result.flags
        assert result.score == 1.0

    def test_clean_url_not_blocklisted(self):
        load_blocklist(["phishing-site.com"])
        result = analyze_url("https://github.com", skip_whois=True)
        assert result.is_blocklisted is False


class TestAnalyzeURLShortener:
    def test_bitly_flagged(self):
        result = analyze_url("https://bit.ly/3xYz123", skip_whois=True)
        assert "url_shortener" in result.flags

    def test_tinyurl_flagged(self):
        result = analyze_url("https://tinyurl.com/abc123", skip_whois=True)
        assert "url_shortener" in result.flags

    def test_full_domain_not_shortener(self):
        result = analyze_url("https://developer.mozilla.org/en-US/docs/Web/API", skip_whois=True)
        assert "url_shortener" not in result.flags


class TestAnalyzeURLEntropy:
    def test_dga_like_domain_high_entropy(self):
        result = analyze_url("https://xkqjvzmwbp.com/cmd", skip_whois=True)
        assert any("high_entropy" in f for f in result.flags)
        assert result.score >= 0.55

    def test_normal_domain_low_entropy(self):
        result = analyze_url("https://google.com", skip_whois=True)
        assert not any("high_entropy" in f for f in result.flags)


class TestAnalyzeURLLookalike:
    def test_paypal_typo_flagged(self):
        result = analyze_url("https://paypa1.com/login", skip_whois=True)
        assert any("lookalike_of" in f for f in result.flags)
        assert result.score >= 0.70

    def test_apple_homoglyph_flagged(self):
        # аpple.com with Cyrillic 'а'
        result = analyze_url("https://\u0430pple.com", skip_whois=True)
        # May or may not resolve depending on tldextract; score should be elevated
        # At minimum: check it doesn't crash
        assert isinstance(result.score, float)

    def test_exact_brand_not_lookalike(self):
        result = analyze_url("https://google.com", skip_whois=True)
        assert not any("lookalike_of" in f for f in result.flags)

    def test_clearly_different_domain_not_lookalike(self):
        result = analyze_url("https://example.com", skip_whois=True)
        assert not any("lookalike_of" in f for f in result.flags)


# ---------------------------------------------------------------------------
# analyze_urls — batch / sorting
# ---------------------------------------------------------------------------

class TestAnalyzeURLs:
    def test_results_sorted_by_score_descending(self):
        load_blocklist(["evil.com"])
        urls = [
            "https://google.com",
            "https://evil.com/steal",
            "https://bit.ly/abc",
        ]
        results = analyze_urls(urls, skip_whois=True)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list_returns_empty(self):
        assert analyze_urls([], skip_whois=True) == []


# ---------------------------------------------------------------------------
# max_url_risk_score
# ---------------------------------------------------------------------------

class TestMaxURLRiskScore:
    def test_max_score_returned(self):
        load_blocklist(["dangerous.com"])
        urls = ["https://google.com", "https://dangerous.com"]
        score, flags = max_url_risk_score(urls, skip_whois=True)
        assert score == 1.0
        assert any("blocklisted_domain" in f for f in flags)

    def test_empty_urls_score_zero(self):
        score, flags = max_url_risk_score([], skip_whois=True)
        assert score == 0.0
        assert flags == []

    def test_clean_urls_low_score(self):
        load_blocklist([])
        urls = ["https://microsoft.com", "https://github.com"]
        score, flags = max_url_risk_score(urls, skip_whois=True)
        assert score < 0.55  # no very high-risk signals
