"""Language detector adapters and runtime measurement."""

from __future__ import annotations

from time import perf_counter


class FastTextDetector:
    """Lazy wrapper around a FastText language ID model."""

    def __init__(self, model_path="lid.176.bin"):
        self.model_path = model_path
        self.model = None

    def _load(self):
        if self.model is None:
            import fasttext

            self.model = fasttext.load_model(self.model_path)
        return self.model

    def predict(self, text, n_results=5):
        """Return `(language_code, probability)` results sorted by probability."""
        model = self._load()
        text = text.replace("\n", " ")
        labels, probs = model.predict(text, k=n_results)
        return [
            (label.replace("__label__", ""), float(prob))
            for label, prob in zip(labels, probs)
        ]


_FASTTEXT_DETECTOR = FastTextDetector()


def detect_resiliparse(text, n_results):
    from resiliparse.parse.lang import detect_fast

    return detect_fast(text, cutoff=2500, n_results=n_results)


def detect_fasttext(text, n_results):
    return _FASTTEXT_DETECTOR.predict(text, n_results=n_results)


def detect_url(text):
    from .url_extractor import detect_url_lang

    return detect_url_lang(text)


def run_model(text, model, n_results):
    """Run one detector and return `(results, runtime_seconds)`."""
    start = perf_counter()
    if model == "rp":
        results = detect_resiliparse(text, n_results)
    elif model == "ft":
        results = detect_fasttext(text, n_results)
    elif model == "url":
        results = detect_url(text)
    else:
        raise ValueError(f"Unknown model: {model}")
    end = perf_counter()
    return results, end - start
