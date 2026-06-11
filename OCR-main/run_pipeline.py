from Postprocessing.process_pipeline import process_ocr_output

if __name__ == "__main__":
    ocr_text = "Ths is a smple txt extrcted frm img."  # Replace with your OCR result
    result = process_ocr_output(ocr_text)
    print("\nFinal Output:\n", result)
