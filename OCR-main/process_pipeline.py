
from .lang_detect import detect_language
from .correct_text import correct_ocr_text
from .translate_text import translate_text

lang_code_map = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml"
}

def process_ocr_output(text):
    print(f"📥 Input OCR Text: {text}")
    
    lang = detect_language(text)
    print(f"🌍 Detected Language: {lang}")
    
    corrected = correct_ocr_text(text)
    print(f"🛠️ Corrected Text: {corrected}")
    
    if lang != "en":
        source_lang = lang_code_map.get(lang, "eng_Latn")
        translated = translate_text(corrected, source_lang)
    else:
        translated = corrected

    print(f"🌐 Translated Text: {translated}")

    return {
        "original_text": text,
        "detected_language": lang,
        "corrected_text": corrected,
        "translated_text": translated
    }
