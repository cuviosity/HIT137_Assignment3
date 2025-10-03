
# ok so this file has the model base class.
from __future__ import annotations
import os, time
from abc import ABC, abstractmethod
from typing import Any

from huggingface_hub import InferenceClient

# --- decorators that i re-use ---
def timed(fn):
    # i like to see how long stuff takes. probably not super accurate but meh.
    def w(*a, **k):
        t0 = time.time()
        out = fn(*a, **k)
        ms = (time.time() - t0) * 1000
        return out, ms
    return w

def ensure_loaded(fn):
    # this just yells at me if i forgot to call .load()
    def w(self, *a, **k):
        if getattr(self, "_client", None) is None:
            raise RuntimeError("uh oh, model not loaded. do self.load() first pls.")
        return fn(self, *a, **k)
    return w

# i wanted logging too but didn't want to import logging module lol
class LoggingMixin:
    def log(self, msg):
        print("[log]", msg)  # TODO: make fancy later maybe

class BaseHFModel(ABC):
    TASK = ""  # i set this in the kids

    def __init__(self, model_id: str):
        # these underscores make it look private
        self._model_id = model_id
        self._client = None
        self._loaded = False

    @property
    def model_id(self):
        return self._model_id

    def load(self):
        token = os.getenv("HF_TOKEN")
        if not token:
            # i keep forgetting this so the message is loud
            raise EnvironmentError("NO HF_TOKEN SET. get one from huggingface and export HF_TOKEN")
        self._client = InferenceClient(model=self._model_id, token=token)
        self._loaded = True
        self.log(f"loaded {self._model_id} (i think)")

    # this is the "template method" thing
    def run(self, input_data: Any):
        stuff = self._preprocess(input_data)
        raw, when_ms = self._predict_timed(stuff)
        nice = self._postprocess(raw)
        return {"output": nice, "latency_ms": round(when_ms, 2), "model_id": self._model_id, "task": self.TASK}

    # the kids have to write these themselves (override!)
    @abstractmethod
    def _preprocess(self, input_data): ...
    @abstractmethod
    def _predict(self, processed): ...
    @abstractmethod
    def _postprocess(self, raw): ...

    # i stack two decorators here
    @timed
    @ensure_loaded
    def _predict_timed(self, processed):
        return self._predict(processed)
