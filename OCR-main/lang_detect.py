
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


LANG_MODEL = "papluca/xlm-roberta-base-language-detection"
tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(LANG_MODEL)

def detect_language(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted = torch.argmax(outputs.logits, dim=1)
    return model.config.id2label[predicted.item()]
