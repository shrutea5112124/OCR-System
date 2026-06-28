Multilingual OCR
Detect and recognize text in multiple scripts, then optionally correct and translate to English.

Supported languages (recognition)
English, Hindi, Gujarati, Chinese, Japanese, Korean, Tamil, Telugu

Setup
pip install -r requirements.txt
Place test images in a test_images/ folder (not tracked in git).

Usage
Full pipeline (OCR + language detection + correction + translation)
# Process a single image
python pipeline/run.py --image test_images/english.jpg

# Process an entire folder and save results
python pipeline/run.py --image test_images/ --output result.json
Evaluation
Measure performance (CER/WER) against ground truth CSVs:

python evaluate.py --csv data/hindi_test.csv --img-dir data/hindi_test_images/ --limit 100
Standalone Preprocessing
Denoise and sharpen an image manually:

from open_cvdenoising import opencv_denoising
clean_img = opencv_denoising("noisy_image.jpg")
Project structure
OCR/
├── pipeline/
│   ├── run.py          # End-to-end CLI
│   └── recognize.py    # OCR wrapper
├── postprocessing/
│   ├── lang_detect.py  # XLM-RoBERTa language detection
│   ├── correct_text.py # FLAN-T5 correction
│   ├── translate_text.py # NLLB translation
│   └── process_pipeline.py
├── rec_multilang.py    # PaddleOCR detect + multi-script recognise
└── model/recognition/  # Per-language OCR modules
Output format
{
  "image": "D:/project_OCR/OCR/test_images/hindi.png",
  "original_text": "...",
  "detected_language": "hi",
  "corrected_text": "...",
  "translated_text": "..."
}
