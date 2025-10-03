
# this is like a tiny factory. i put strings here so the dropdown has friendly names.
from .text_sentiment import TextSentimentModel
from .image_classification import ImageClassificationModel

REGISTRY = {
    "Text: Sentiment (easy one)": TextSentimentModel,
    "Vision: Image Classifier (the picture one)": ImageClassificationModel,
}

def get_model_names():
    return list(REGISTRY.keys())

def create_model(name):
    return REGISTRY[name]()  # fingers crossed
