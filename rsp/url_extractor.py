"""URL-based language hints for detector output generation."""

from __future__ import annotations

from urllib.parse import urlparse


SUPPORTED_LANGS = {
    "af", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "ca",
    "ce", "cs", "cv", "cy", "da", "de", "dv", "el", "en", "eo", "es",
    "et", "eu", "fa", "fi", "fo", "fr", "fy", "ga", "gd", "gl", "gu",
    "ha", "he", "hi", "hr", "hu", "hy", "id", "io", "is", "it", "ja",
    "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", "la", "lb", "lt",
    "lv", "mg", "mk", "ml", "mn", "mr", "mt", "my", "ne", "nl", "no",
    "or", "pa", "pl", "ps", "pt", "rm", "ro", "ru", "sa", "sc", "sd",
    "si", "sk", "sl", "so", "sq", "sr", "sv", "sw", "ta", "te", "tg",
    "th", "tk", "tl", "tr", "tt", "ug", "uk", "ur", "uz", "vi", "vo",
    "yi", "zh",
}

TLD_TO_LANG = {
    "de": "de",
    "fr": "fr",
    "it": "it",
    "es": "es",
    "pt": "pt",
    "nl": "nl",
    "pl": "pl",
    "ru": "ru",
    "cz": "cs",
    "se": "sv",
    "no": "no",
    "fi": "fi",
    "tr": "tr",
    "uk": "uk",
    "cn": "zh",
    "jp": "ja",
    "kr": "ko",
    "gr": "el",
    "dk": "da",
    "is": "is",
    "lt": "lt",
    "lv": "lv",
    "ee": "et",
    "hu": "hu",
    "ro": "ro",
    "bg": "bg",
    "hr": "hr",
    "sk": "sk",
    "si": "sl",
    "rs": "sr",
    "ba": "sr",
    "vn": "vi",
    "th": "th",
    "id": "id",
    "il": "he",
}


def detect_url_lang(url: str):
    """Return a `(language, confidence)` hint from URL subdomain or TLD."""
    suffix, subdomain = extract_url_parts(url)

    if subdomain:
        prefix = subdomain.split(".")[0]
        if prefix in SUPPORTED_LANGS:
            return prefix, 0.95

    if suffix in TLD_TO_LANG:
        return TLD_TO_LANG[suffix], 0.80

    return "unknown", 0.0


def extract_url_parts(url: str) -> tuple[str, str]:
    """Extract `(suffix, subdomain)` with tldextract when available."""
    try:
        import tldextract
    except ModuleNotFoundError:
        parsed = urlparse(url if "://" in url else f"//{url}")
        host = (parsed.hostname or "").lower()
        parts = host.split(".")
        if len(parts) < 2:
            return "", ""
        suffix = parts[-1]
        subdomain = ".".join(parts[:-2])
        return suffix, subdomain

    ext = tldextract.extract(url)
    return (ext.suffix or "").lower(), (ext.subdomain or "").lower()
