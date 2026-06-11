from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


TRANSLATE_MODEL = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL)

def translate_text(text, source_lang="hin_Deva", target_lang="eng_Latn"):
    inputs = tokenizer(text, return_tensors="pt")
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
