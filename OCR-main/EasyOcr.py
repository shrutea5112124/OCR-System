import easyocr
import os

# ✅ Paths
input_folder = r"D:\Poly OCR\OCR\test images"
output_folder = r"D:\Poly OCR\OCR\recognized_text"
os.makedirs(output_folder, exist_ok=True)

# ✅ Language suffix → EasyOCR language list
lang_map = {
    'en': ['en'],
    'hi': ['hi', 'en'],
    'ch': ['ch_sim', 'en'],
    'ja': ['ja', 'en'],
    'ko': ['ko', 'en'],
    'te': ['te', 'en'],
    'fr': ['fr', 'en'],
}

# ✅ Supported image extensions
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# ✅ Loop through each image in the folder
for filename in os.listdir(input_folder):
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        continue

    print(f"\n🔍 Processing: {filename}")
    image_path = os.path.join(input_folder, filename)

    # ✅ Extract language from filename: expects _lang.jpg (e.g. img_fr.jpg)
    try:
        lang_key = filename.split('_')[-1].split('.')[0].lower()
        lang_list = lang_map[lang_key]
    except KeyError:
        print(f"⚠️ Skipping '{filename}': Unknown or unsupported language code.")
        continue

    # ✅ Initialize EasyOCR reader
    try:
        reader = easyocr.Reader(lang_list=lang_list, gpu=False)
    except Exception as e:
        print(f"❌ Error initializing EasyOCR for {filename}: {e}")
        continue

    # ✅ Perform OCR
    try:
        results = reader.readtext(image_path)
    except Exception as e:
        print(f"❌ OCR failed on {filename}: {e}")
        continue

    # ✅ Print and save recognized text
    recognized_lines = []
    for (_, text, conf) in results:
        print(f"📝 {text}")
        recognized_lines.append(text)

    out_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in recognized_lines:
            f.write(line + "\n")



