"""Source credibility scoring based on domain authority and other factors."""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _normalized_host(url: str) -> str:
    """Lowercase host without default ports; strip leading www."""
    if not url or not isinstance(url, str):
        return ""
    try:
        netloc = urlparse(url.strip()).netloc.lower()
    except Exception:
        return ""
    if not netloc:
        return ""
    for suffix in (":443", ":80"):
        if netloc.endswith(suffix):
            netloc = netloc[: -len(suffix)]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _extract_url(result: Any) -> str:
    """Tavily row dict, object with .url, or raw string."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, Mapping):
        u = result.get("url")
        return u.strip() if isinstance(u, str) else ""
    url = getattr(result, "url", None)
    return url.strip() if isinstance(url, str) else ""


def _matches_trusted(host: str, trusted: str) -> bool:
    """
    trusted: either a public suffix like '.edu' / '.ac.uk',
    or a registrable hostname like 'reuters.com' (subdomains allowed).
    """
    trusted = trusted.lower().strip()
    if not trusted or not host:
        return False
    if trusted.startswith("."):
        return host.endswith(trusted)
    return host == trusted or host.endswith("." + trusted)


class CredibilityScorer:
    """Score sources based on domain authority and other credibility factors."""

    TRUSTED_DOMAINS: frozenset[str] = frozenset(
        {
            ".edu",
            ".ac.uk",
            ".ac.in",
            ".edu.in",
            ".edu.au",
            ".ac.jp",
            ".gov",
            ".gov.uk",
            ".gov.au",
            ".gov.ca",
            ".gov.in",
            ".europa.eu",
            "bbc.com",
            "bbc.co.uk",
            "reuters.com",
            "ap.org",
            "npr.org",
            "theguardian.com",
            "nytimes.com",
            "washingtonpost.com",
            "wsj.com",
            "ft.com",
            "economist.com",
            "bloomberg.com",
            "cnbc.com",
            "cnn.com",
            "aljazeera.com",
            "france24.com",
            "dw.com",
            "thehindu.com",
            "indianexpress.com",
            "timesofindia.com",
            "indiatimes.com",
            "economictimes.com",
            "financialexpress.com",
            "livemint.com",
            "business-standard.com",
            "moneycontrol.com",
            "businessline.in",
            "businesstoday.in",
            "businessinsider.in",
            "arxiv.org",
            "scholar.google.com",
            "researchgate.net",
            "semanticscholar.org",
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov",
            "nih.gov",
            "nature.com",
            "sciencedirect.com",
            "springer.com",
            "wiley.com",
            "ieee.org",
            "jstor.org",
            "plos.org",
            "sciencemag.org",
            "cell.com",
            "who.int",
            "cdc.gov",
            "mayoclinic.org",
            "webmd.com",
            "un.org",
            "worldbank.org",
            "imf.org",
            "wto.org",
            "oecd.org",
            "scientificamerican.com",
            "newscientist.com",
            "technologyreview.com",
            "spectrum.ieee.org",
            "arstechnica.com",
            "wired.com",
            "techcrunch.com",
            "theverge.com",
            "wikipedia.org",
            "britannica.com",
            "khanacademy.org",
            "supremecourt.gov",
            "congress.gov",
            "loc.gov",
            "census.gov",
            "bls.gov",
            "data.gov",
            "statista.com",
            "pewresearch.org",
            "gallup.com",
        }
    )

    SUSPICIOUS_HOST_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\.(xyz|tk|ml|ga|cf|gq)$", re.IGNORECASE),
        re.compile(r"(^|\.)bit\.ly$|(^|\.)tinyurl\.|(^|\.)t\.co$", re.IGNORECASE),
        re.compile(r"blogspot\.|wordpress\.com$", re.IGNORECASE),
    )

    def score_url(self, url: str) -> dict[str, Any]:
        if not url or not isinstance(url, str):
            return {"score": 0, "factors": ["No URL"], "level": "low", "domain": ""}

        score = 50
        factors: list[str] = []

        try:
            parsed = urlparse(url.strip())
            domain = _normalized_host(url)
            if not domain:
                return {"score": 0, "factors": ["Invalid URL"], "level": "low", "domain": ""}

            is_trusted = False
            for trusted in self.TRUSTED_DOMAINS:
                if _matches_trusted(domain, trusted):
                    score += 30
                    factors.append(f"Trusted domain ({trusted})")
                    is_trusted = True
                    break

            is_suspicious = False
            for pat in self.SUSPICIOUS_HOST_PATTERNS:
                if pat.search(domain):
                    score -= 20
                    factors.append(f"Suspicious host pattern: {pat.pattern}")
                    is_suspicious = True
                    break

            if parsed.scheme == "https":
                score += 5
                factors.append("HTTPS")
            elif parsed.scheme == "http":
                score -= 10
                factors.append("HTTP only")

            if not is_trusted and not is_suspicious:
                if len(domain.split(".")) > 3:
                    score -= 5
                    factors.append("Deep subdomain chain")

            path_lower = (parsed.path or "").lower()
            if any(
                p in path_lower
                for p in ("/papers/", "/research/", "/publications/", "/article/", "/doi/")
            ):
                score += 10
                factors.append("Research-style path")

            score = max(0, min(100, score))

            if score >= 70:
                level = "high"
            elif score >= 40:
                level = "medium"
            else:
                level = "low"

            return {
                "score": score,
                "factors": factors if factors else ["Standard domain"],
                "level": level,
                "domain": domain,
            }

        except Exception as e:
            logger.warning("Error scoring URL %s: %s", url, e)
            return {"score": 30, "factors": ["Scoring error"], "level": "low", "domain": ""}

    def score_search_results(self, results: Sequence[Any]) -> list[dict[str, Any]]:
        """Attach credibility to each item; sort by score descending."""
        scored: list[dict[str, Any]] = []
        for result in results:
            url = _extract_url(result)
            credibility = self.score_url(url)
            scored.append({"result": result, "url": url, "credibility": credibility})

        scored.sort(key=lambda x: x["credibility"]["score"], reverse=True)
        return scored

    def filter_by_credibility(
        self,
        results: Sequence[Any],
        min_score: int = 40,
        *,
        ensure_at_least_one: bool = True,
    ) -> list[Any]:
        """
        Drop rows below min_score. If none pass and ensure_at_least_one is True,
        keep the single highest-scoring row (scored list is already sorted desc).
        """
        scored = self.score_search_results(results)
        filtered = [item["result"] for item in scored if item["credibility"]["score"] >= min_score]
        if not filtered and scored and ensure_at_least_one:
            logger.warning(
                "All results below min_score=%s; keeping top-scoring URL only",
                min_score,
            )
            filtered = [scored[0]["result"]]
        logger.info("Filtered %s -> %s results (min_score=%s)", len(results), len(filtered), min_score)
        return filtered

    def annotate_tavily_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """
        Return new dicts with original keys plus 'credibility' and '_credibility_score'.
        Safe for read-only Tavily dicts (copies each row).
        """
        out: list[dict[str, Any]] = []
        for row in rows:
            url = _extract_url(row)
            cred = self.score_url(url)
            merged = {**dict(row), "credibility": cred, "_credibility_score": cred["score"]}
            out.append(merged)
        out.sort(key=lambda r: r.get("_credibility_score", 0), reverse=True)
        return out


def rank_and_filter_search_results(
    results: Sequence[Any],
    min_credibility_score: int = 40,
    scorer: CredibilityScorer | None = None,
    *,
    ensure_at_least_one: bool = True,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Score -> sort high first -> filter by min score.

    Returns (filtered_results, credibility_rows) aligned by index.

    If every row is below min_credibility_score and ensure_at_least_one is True,
    keeps only the single highest-scoring row (never returns an empty list when
    results was non-empty). Set ensure_at_least_one=False to allow zero rows.
    """
    scorer = scorer or CredibilityScorer()
    scored = scorer.score_search_results(results)
    kept = [x for x in scored if x["credibility"]["score"] >= min_credibility_score]

    if not kept and scored and ensure_at_least_one:
        logger.warning(
            "All results below min_credibility_score=%s; keeping top-scoring row only",
            min_credibility_score,
        )
        kept = [scored[0]]

    filtered_results = [x["result"] for x in kept]
    credibility_rows = [x["credibility"] for x in kept]
    return filtered_results, credibility_rows
