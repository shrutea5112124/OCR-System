#  Multilingual OCR System

A deep learning-based Optical Character Recognition (OCR) system capable of detecting, recognizing, and classifying multilingual text from images. This project combines Computer Vision, Deep Learning, and Natural Language Processing techniques to build an end-to-end OCR pipeline.

---

## Project Overview

The goal of this project is to extract text from images containing multiple languages and provide accurate recognition through image preprocessing, text detection, text recognition, language classification, and post-processing techniques.

The system is designed to handle:

- Scanned documents
- Natural scene images
- Signboards
- Printed multilingual text
- Low-quality and noisy images

---

## Features

### Image Preprocessing
- Image resizing
- Grayscale conversion
- Noise removal
- Adaptive thresholding
- Contrast enhancement
- Image normalization

### Text Detection
- Detection of text regions
- Bounding box generation
- Multi-language text localization

### Text Recognition
- Character recognition
- Word recognition
- Deep learning-based OCR pipeline

### Language Classification
- Automatic language identification
- CNN-based language classifier
- Support for multiple languages

### Post Processing
- Text cleaning
- Formatting correction
- Error handling

### Spell Correction
- Dictionary-based correction
- Context-aware correction

### Translation Support
- Optional translation of extracted text

---

## System Architecture

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
Text Detection
     │
     ▼
Text Recognition
     │
     ▼
Language Classification
     │
     ▼
Post Processing
     │
     ▼
Spell Correction
     │
     ▼
Final Output
```

---

##  Project Structure

```text
OCR-System/
│
├── dataset/
│   ├── TrainImages/
│   ├── TrainGT/
│   └── TestImages/
│
├── preprocessing/
│
├── detection/
│
├── recognition/
│
├── language_classifier/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── postprocessing/
│
├── outputs/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

##  Dataset

This project uses the **ICDAR MLT 2019 Dataset** for multilingual text recognition and language classification.

### Dataset Structure

```text
dataset/
│
├── TrainImages/
├── TrainGT/
└── TestImages/
```

### Supported Languages

- English
- Arabic
- Bangla
- Chinese
- French
- German
- Hindi
- Italian
- Japanese
- Korean

---

## Technologies Used

### Programming Language
- Python

### Deep Learning
- PyTorch
- Torchvision

### Computer Vision
- OpenCV
- Pillow

### OCR Libraries
- PaddleOCR
- Tesseract OCR

### Data Processing
- NumPy
- Pandas

### Web Framework
- Flask

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/OCR-System.git
cd OCR-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Training

Train the language classification model:

```bash
python train.py
```

The training pipeline:

1. Loads ICDAR MLT 2019 dataset
2. Extracts language labels
3. Trains CNN classifier
4. Saves trained model weights

---

##  Running OCR

```bash
python main.py
```

OCR Pipeline:

1. Load image
2. Preprocess image
3. Detect text regions
4. Recognize text
5. Identify language
6. Correct spelling
7. Generate final output

---

##  Example Output

### Input

Image containing multilingual text.

### Output

```text
Detected Text:
नमस्ते दुनिया

Language:
Hindi

Corrected Text:
नमस्ते दुनिया

English Translation:
Hello World
```

---

##  Applications

- Document Digitization
- Historical Archive Preservation
- Smart Scanning Systems
- Translation Tools
- Government Document Processing
- Educational Applications
- Multilingual Information Extraction

---

##  Future Improvements

- Handwritten Text Recognition
- Transformer-Based OCR Models
- Real-Time Camera OCR
- Mobile Application
- Cloud Deployment
- LLM-Based Contextual Correction
- PDF OCR Support

---

##  License

This project is licensed under the MIT License.

---

##  Author

**Shruti**

AI/ML Enthusiast | Computer Vision | OCR Systems

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub!
