
# TEXT MODEL (sentiment). i just used the sst2 one because everyone does.
from typing import Any
from .base import BaseHFModel, LoggingMixin

class TextSentimentModel(LoggingMixin, BaseHFModel):
    TASK = "text-classification"

    def __init__(self, model_id="distilbert-base-uncased-finetuned-sst-2-english"):
        super().__init__(model_id)

    def _preprocess(self, input_data: Any):
        if not isinstance(input_data, str) or not input_data.strip():
            raise ValueError("pls type some text first")
        txt = input_data.strip()
        self.log(f"got text len={len(txt)}")
        return txt

    def _predict(self, processed):
        # hosted api call, no downloads (yay)
        self.log("sending to hf (text-classification) ...")
        return self._client.text_classification(processed)

    def _postprocess(self, raw):
        if not raw:
            return "no idea sorry"
        best = max(raw, key=lambda r: r.get("score", 0))
        return f"guess: {best.get('label','?')} (score {best.get('score',0):.3f})"
