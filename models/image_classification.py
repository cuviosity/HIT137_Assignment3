
# IMAGE MODEL (classification). i picked a ViT because it sounded cool
from io import BytesIO
from PIL import Image
from .base import BaseHFModel, LoggingMixin

class ImageClassificationModel(LoggingMixin, BaseHFModel):
    TASK = "image-classification"

    def __init__(self, model_id="google/vit-base-patch16-224"):
        super().__init__(model_id)

    def _preprocess(self, input_data):
        # i accept either path or bytes. probably could be better.
        if isinstance(input_data, (bytes, bytearray)):
            data = bytes(input_data)
        else:
            if not isinstance(input_data, str):
                raise ValueError("pls give me an image path (png/jpg)")
            im = Image.open(input_data).convert("RGB")
            buf = BytesIO()
            im.save(buf, format="PNG")
            data = buf.getvalue()
        self.log(f"image bytes = {len(data)}")
        return data

    def _predict(self, processed):
        self.log("sending to hf (image-classification) ...")
        return self._client.image_classification(processed)

    def _postprocess(self, raw):
        if not raw:
            return "no classes?? weird"
        best = max(raw, key=lambda r: r.get("score", 0))
        return f"top: {best.get('label','?')} ({best.get('score',0):.3f})"
