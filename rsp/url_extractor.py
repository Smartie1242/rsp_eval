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


def detect_url_lang(url: str) -> tuple[str, float]:
    """
    Return a (language, confidence) hint from URL structure.

    Detection order:
        1. Language subdomain (fy.example.com)
        2. Language path segment (/fy/, /fy-NL/, /zh-Hans/)
        3. Country-code TLD (.nl, .de, .fr)
        4. Fallback to English
    """
    parsed = urlparse(url if "://" in url else f"//{url}")

    host = (parsed.hostname or "").lower()
    parts = host.split(".")

    if len(parts) >= 3:
        prefix = parts[0]
        if prefix in SUPPORTED_LANGS:
            return prefix, 0.95

    path_lang = extract_path_language(parsed.path)
    if path_lang:
        return path_lang, 0.90

    if len(parts) >= 2:
        lang = TLD_TO_LANG.get(parts[-1])
        if lang:
            return lang, 0.80

    return "en", 0.05


def extract_path_language(path: str) -> str:
    """
    Detect language codes from path segments.
    """
    for segment in path.lower().split("/"):
        if len(segment) < 2:
            continue

        lang = segment[:2]

        if lang not in SUPPORTED_LANGS:
            continue

        if len(segment) == 2:
            return lang

        if segment[2:3] in ("-", "_"):
            return lang

    return ""