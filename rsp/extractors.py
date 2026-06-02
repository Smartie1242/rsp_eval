"""Dataset row extractors for supported research datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .languages import OWI_LABEL_TO_ISO3, normalize_to_iso3


class DatasetExtractor(ABC):
    """Abstract base class for extracting text, labels, URLs, and metadata."""

    @abstractmethod
    def extract_text(self, row: Dict) -> str:
        """Extract text content from a dataset row."""

    @abstractmethod
    def extract_label(self, row: Dict) -> Optional[str]:
        """Extract the ground-truth language label from a dataset row."""

    def extract_url(self, row: Dict) -> Optional[str]:
        """Extract a URL if available."""
        return ""

    def extract_metadata(self, row: Dict) -> Dict[str, str]:
        """Extract optional metadata fields for CSV output."""
        return {}

    def metadata_headers(self) -> List[str]:
        """Return additional metadata header names."""
        return []


class CommonLIDExtractor(DatasetExtractor):
    """Extractor for CommonLID rows with `tag` and `text` fields."""

    def extract_text(self, row: Dict) -> str:
        text = row.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\x00", "")
        text = text.replace("\ufeff", "")
        text = text.replace("\r", " ").replace("\n", " ")
        text = text.encode("utf-8", "replace").decode("utf-8")
        return " ".join(text.split())

    def extract_label(self, row: Dict) -> Optional[str]:
        return normalize_to_iso3(row.get("tag"))

    def extract_url(self, row: Dict) -> Optional[str]:
        return ""

    def extract_metadata(self, row: Dict) -> Dict[str, str]:
        return {"tag": row.get("tag", "")}

    def metadata_headers(self) -> List[str]:
        return ["tag"]


class WiLIExtractor(DatasetExtractor):
    """Extractor for WiLI-2018 records."""

    def extract_text(self, row: Dict) -> str:
        return row.get("text", "")

    def extract_label(self, row: Dict) -> Optional[str]:
        return normalize_to_iso3(row.get("label"))

    def extract_url(self, row: Dict) -> Optional[str]:
        return ""

    def extract_metadata(self, row: Dict) -> Dict[str, str]:
        return {
            "split": row.get("split", "") if row.get("split") else "test",
            "source_index": str(row.get("source_index", "")),
        }

    def metadata_headers(self) -> List[str]:
        return ["split", "source_index"]


class WikiExtractor(DatasetExtractor):
    """Extractor for wiki datasets stored in language subdirectories."""

    def extract_text(self, row: Dict) -> str:
        return row.get("text", "")

    def extract_label(self, row: Dict) -> Optional[str]:
        return normalize_to_iso3(row.get("label"))

    def extract_url(self, row: Dict) -> Optional[str]:
        return ""

    def extract_metadata(self, row: Dict) -> Dict[str, str]:
        metadata = {
            "source_file": row.get("source_file", ""),
            "doc_id": row.get("doc_id", ""),
        }
        if row.get("language"):
            metadata["language"] = row.get("language")
        return metadata

    def metadata_headers(self) -> List[str]:
        return ["doc_id", "language", "source_file"]


class OWIExtractor(DatasetExtractor):
    """Extractor for Label Studio annotated OWI datasets."""

    def __init__(self):
        self.mapping = OWI_LABEL_TO_ISO3

    def extract_text(self, row: Dict) -> str:
        return row.get("data", {}).get("text", "")

    def extract_label(self, row: Dict) -> Optional[str]:
        try:
            annotations = row.get("annotations", [])
            if not annotations:
                return None

            results = annotations[0].get("result", [])
            if not results:
                return None

            choices = results[0].get("value", {}).get("choices", [])
            if not choices:
                return None

            return self.mapping.get(choices[0])
        except (KeyError, IndexError, TypeError):
            return None

    def extract_url(self, row: Dict) -> str:
        return row.get("data", {}).get("url", "")

    def extract_metadata(self, row: Dict) -> Dict[str, str]:
        data = row.get("data", {})
        return {
            "doc_id": data.get("doc_id", ""),
            "raw_text": data.get("raw_text", ""),
            "clean_text": data.get("clean_text", ""),
            "raw_length": str(data.get("raw_length", "")),
            "clean_length": str(data.get("clean_length", "")),
            "source": data.get("source", ""),
            "url": data.get("url", ""),
            "title": data.get("title", ""),
        }

    def metadata_headers(self) -> List[str]:
        return [
            "doc_id",
            "raw_text",
            "clean_text",
            "raw_length",
            "clean_length",
            "source",
            "url",
            "title",
        ]


class SimpleExtractor(DatasetExtractor):
    """Extractor for records with direct `text` and `label` fields."""

    def extract_text(self, row: Dict) -> str:
        return row.get("text", "")

    def extract_label(self, row: Dict) -> Optional[str]:
        return row.get("label")


class CustomExtractor(DatasetExtractor):
    """Flexible extractor using callables for text, label, and metadata."""

    def __init__(self, text_fn, label_fn, metadata_fn=None, metadata_headers=None):
        self.text_fn = text_fn
        self.label_fn = label_fn
        self.metadata_fn = metadata_fn
        self._metadata_headers = metadata_headers or []

    def extract_text(self, row: Dict) -> str:
        return self.text_fn(row)

    def extract_label(self, row: Dict) -> Optional[str]:
        return self.label_fn(row)

    def extract_metadata(self, row: Dict) -> Dict[str, str]:
        if self.metadata_fn is None:
            return {}
        return self.metadata_fn(row)

    def metadata_headers(self) -> List[str]:
        return self._metadata_headers


def get_extractor(extractor_type: str, **kwargs) -> DatasetExtractor:
    """Return an extractor instance by name."""
    if extractor_type == "commonlid":
        return CommonLIDExtractor()
    if extractor_type == "owi":
        return OWIExtractor()
    if extractor_type == "wili":
        return WiLIExtractor()
    if extractor_type == "wiki":
        return WikiExtractor()
    if extractor_type == "simple":
        return SimpleExtractor()
    if extractor_type == "custom":
        if "text_fn" not in kwargs or "label_fn" not in kwargs:
            raise ValueError("custom extractor requires 'text_fn' and 'label_fn'")
        return CustomExtractor(
            kwargs["text_fn"],
            kwargs["label_fn"],
            metadata_fn=kwargs.get("metadata_fn"),
            metadata_headers=kwargs.get("metadata_headers"),
        )
    raise ValueError(f"Unknown extractor type: {extractor_type}")

