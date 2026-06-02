"""Language code normalization and distance helpers."""

from __future__ import annotations

from typing import Optional

LANGUAGE_MAP = {
    "Frisian": "fy",
    "Dutch": "nl",
    "German": "de",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Chinese": "zh",
    "Russian": "ru",
    "Portuguese": "pt",
    "Italian": "it",
    "Arabic": "ar",
    "Japanese": "ja",
    "Turkish": "tr",
    "Indonesian": "id",
    "Polish": "pl",
    "Persian": "fa",
    "Hebrew": "he",
    "Vietnamese": "vi",
    "Swedish": "sv",
    "Korean": "ko",
    "Hindi": "hi",
    "Ukrainian": "uk",
    "Romanian": "ro",
    "Czech": "cs",
    "Norwegian": "no",
    "Finnish": "fi",
    "Hungarian": "hu",
    "Danish": "da",
    "Thai": "th",
    "Catalan": "ca",
    "Bengali": "bn",
    "Greek": "el",
    "Bulgarian": "bg",
    "Serbian": "sr",
    "Croatian": "hr",
    "Azerbaijani": "az",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Tamil": "ta",
    "Esperanto": "eo",
    "Lithuanian": "lt",
    "Estonian": "et",
    "Malayalam": "ml",
    "Latin": "la",
    "Afrikaans": "af",
    "Marathi": "mr",
    "Albanian": "sq",
    "Urdu": "ur",
    "Georgian": "ka",
    "Basque": "eu",
    "Galician": "gl",
    "Tagalog": "tl",
    "Armenian": "hy",
    "Kazakh": "kk",
    "Belarusian": "be",
    "Telugu": "te",
    "Latvian": "lv",
    "Macedonian": "mk",
    "Burmese": "my",
    "Icelandic": "is",
    "Mongolian": "mn",
    "Kannada": "kn",
    "Welsh": "cy",
    "Breton": "br",
    "Uzbek": "uz",
    "Gujarati": "gu",
    "Nepali": "ne",
    "Sinhala": "si",
    "Luxembourgish": "lb",
    "Javanese": "jv",
    "Swahili": "sw",
    "Irish": "ga",
    "Kurdish": "ku",
    "Yiddish": "yi",
    "Tatar": "tt",
    "Punjabi": "pa",
    "Khmer": "km",
    "Tajik": "tg",
    "Sanskrit": "sa",
    "Bashkir": "ba",
    "Ido": "io",
    "Assamese": "as",
    "Volapük": "vo",
    "Kyrgyz": "ky",
    "Somali": "so",
    "Chuvash": "cv",
    "Odia": "or",
    "Chechen": "ce",
    "Malagasy": "mg",
    "Pashto": "ps",
    "Faroese": "fo",
    "Tibetan": "bo",
    "Scottish Gaelic": "gd",
    "Turkmen": "tk",
    "Divehi": "dv",
    "Sardinian": "sc",
    "Maltese": "mt",
    "Uyghur": "ug",
    "Romansh": "rm",
    "Hausa": "ha",
    "Sindhi": "sd",
    "Other": "other",
    "Mixed languages": "mixed",
    "Unknown": "unknown",
}

ISO_MAP = {
    "de": "deu",
    "fy": "fry",
    "nl": "nld",
}

SPECIAL_LANGUAGE_MAPPINGS = {
    "dut": "nld",
    "ger": "deu",
    "fre": "fra",
    "gre": "ell",
    "rum": "ron",
    "per": "fas",
    "bur": "mya",
    "arm": "hye",
    "baq": "eus",
    "tib": "bod",
    "wel": "cym",
    "slo": "slk",
    "alb": "sqi",
    "ice": "isl",
    "mac": "mkd",
}

OWI_LABEL_TO_ISO3 = {
    "Frisian": "fry",
    "Dutch": "nld",
    "German": "deu",
    "English": "eng",
    "Spanish": "spa",
    "French": "fra",
    "Chinese": "zho",
    "Russian": "rus",
    "Portuguese": "por",
    "Italian": "ita",
    "Arabic": "ara",
    "Japanese": "jpn",
    "Turkish": "tur",
    "Indonesian": "ind",
    "Polish": "pol",
    "Persian": "fas",
    "Hebrew": "heb",
    "Vietnamese": "vie",
    "Swedish": "swe",
    "Korean": "kor",
    "Hindi": "hin",
    "Ukrainian": "ukr",
    "Romanian": "ron",
    "Czech": "ces",
    "Norwegian": "nor",
    "Finnish": "fin",
    "Hungarian": "hun",
    "Danish": "dan",
    "Thai": "tha",
    "Catalan": "cat",
    "Bengali": "ben",
    "Greek": "ell",
    "Bulgarian": "bul",
    "Serbian": "srp",
    "Croatian": "hrv",
    "Azerbaijani": "aze",
    "Slovak": "slk",
    "Slovenian": "slv",
    "Tamil": "tam",
    "Esperanto": "epo",
    "Lithuanian": "lit",
    "Estonian": "est",
    "Malayalam": "mal",
    "Latin": "lat",
    "Afrikaans": "afr",
    "Marathi": "mar",
    "Albanian": "sqi",
    "Urdu": "urd",
    "Georgian": "kat",
    "Basque": "eus",
    "Galician": "glg",
    "Tagalog": "tgl",
    "Armenian": "hye",
    "Kazakh": "kaz",
    "Belarusian": "bel",
    "Telugu": "tel",
    "Latvian": "lav",
    "Macedonian": "mkd",
    "Burmese": "mya",
    "Icelandic": "isl",
    "Mongolian": "mon",
    "Kannada": "kan",
    "Welsh": "cym",
    "Breton": "bre",
    "Uzbek": "uzb",
    "Gujarati": "guj",
    "Nepali": "nep",
    "Sinhala": "sin",
    "Luxembourgish": "ltz",
    "Javanese": "jav",
    "Swahili": "swa",
    "Irish": "gle",
    "Kurdish": "kur",
    "Yiddish": "yid",
    "Tatar": "tat",
    "Punjabi": "pan",
    "Khmer": "khm",
    "Tajik": "tgk",
    "Sanskrit": "san",
    "Bashkir": "bak",
    "Ido": "ido",
    "Assamese": "asm",
    "Volapük": "vol",
    "Kyrgyz": "kir",
    "Somali": "som",
    "Chuvash": "chv",
    "Odia": "ori",
    "Chechen": "che",
    "Malagasy": "mlg",
    "Pashto": "pus",
    "Faroese": "fao",
    "Tibetan": "bod",
    "Scottish Gaelic": "gla",
    "Turkmen": "tuk",
    "Divehi": "div",
    "Sardinian": "srd",
    "Maltese": "mlt",
    "Uyghur": "uig",
    "Romansh": "roh",
    "Hausa": "hau",
    "Sindhi": "snd",
    "Other": "other",
    "Mixed languages": "mixed",
    "Unknown": "unknown",
}


def normalize_language(label: str) -> Optional[str]:
    """Convert a full language name to its ISO-639-1 code when known."""
    return LANGUAGE_MAP.get(label, label)


def get_iso_code(lang_code: str) -> str:
    """Convert selected ISO-639-1 codes to ISO-639-3 codes."""
    return ISO_MAP.get(lang_code, lang_code)


def normalize_to_iso3(code: Optional[str]) -> Optional[str]:
    """Normalize ISO-639-1, ISO-639-3, BCP-47, or CommonLID tags to ISO-639-3."""
    if not code:
        return None

    code = str(code).strip().lower()
    if code in SPECIAL_LANGUAGE_MAPPINGS:
        return SPECIAL_LANGUAGE_MAPPINGS[code]

    base = code.split("_")[0].split("-")[0]

    try:
        import pycountry
    except ImportError as exc:
        raise ImportError("normalize_to_iso3 requires pycountry") from exc

    lang = pycountry.languages.get(alpha_3=base)
    if lang and hasattr(lang, "alpha_3"):
        return lang.alpha_3

    lang = pycountry.languages.get(alpha_2=base)
    if lang and hasattr(lang, "alpha_3"):
        return lang.alpha_3

    return None


def safe_tag_distance(a, b):
    """Compute a language tag distance, returning a large value for invalid tags."""
    bad_values = {"unknown", "unk", "none", "", None}
    if a in bad_values or b in bad_values:
        return 100

    try:
        from langcodes import tag_distance

        return tag_distance(str(a), str(b))
    except Exception:
        return 100

