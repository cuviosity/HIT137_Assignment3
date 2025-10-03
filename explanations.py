
# May be slightly messy, can streamline?

OOP_EXPLANATIONS = {
    "multiple_inheritance": "my classes inherit from BaseHFModel AND LoggingMixin (two for one).",
    "encapsulation": "i hid stuff like _client and _model_id because i heard that was good practice.",
    "polymorphism": "both models have .run(x) but they do different things (text vs image), which is neat.",
    "method_overriding": "the children change _preprocess/_predict/_postprocess to suit their vibe.",
    "multiple_decorators": "stacked @ensure_loaded and @timed because sometimes i forget to load and i like timings."
}
