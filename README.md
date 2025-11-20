#  Evoastra Image Captioning System

### **AI-powered Image Caption Generator using BLIP + FastAPI + HTML/JS UI**

The Evoastra Image Captioning System is a complete end-to-end application that generates **human-like captions** for images using a state-of-the-art **Vision–Language Transformer (BLIP)**.
It also computes a **BLEU-based confidence score** to measure caption stability.


#  Features

* ✔ **Pretrained BLIP model** for high-quality captioning
* ✔ **FastAPI backend** for fast inference
* ✔ **Modern frontend UI** with preview, spinner, and caption display
* ✔ **BLEU confidence score** to evaluate caption reliability
* ✔ Supports CPU (no GPU required)
* ✔ No training required
* ✔ Lightweight and easy to deploy

---

#  Model Used

### **1️⃣ BLIP (Bootstrapping Language-Image Pretraining)**

* Model: `Salesforce/blip-image-captioning-base`
* Type: Vision–Language Transformer
* Encoder: Vision Transformer (ViT)
* Decoder: GPT-like text generator
* Reason: High accuracy, zero training needed, works well on CPU

### **2️⃣ Prior Attempted Model (Not used in final)**

* CNN–LSTM architecture
* Encoder: ResNet-50
* Decoder: LSTM
* Replaced due to slow training and lower accuracy

---

#  Evaluation Metric Used

###  **BLEU Score (Bilingual Evaluation Understudy)**

* Standard metric in image captioning
* Measures similarity between two sentences
* BLEU = 0 → completely different
* BLEU = 1 → identical

### Why BLEU Works Here

Since uploaded images have **no ground truth caption**, we use:
✔ **Caption A** – greedy decoding
✔ **Caption B** – sampling-based decoding

Then compute BLEU(A, B) to measure **caption stability**.

Interpretation:

| BLEU Range | Confidence |
| ---------- | ---------- |
| 0.8 – 1.0  | High       |
| 0.5 – 0.8  | Moderate   |
| < 0.5      | Low        |

---

#  Project Folder Structure

```
MAJOR PROJECT/
│── backend/
│   ├── caption_api.py          # FastAPI backend with BLIP + BLEU
│   ├── other backend files
│
├── index.html                  # Complete frontend UI
├── .gitignore                  # Ignores dataset, venv, cache, model files
├── README.md                   # Project documentation
└── venv/                       # Virtual environment (ignored)
```

---

#  Installation

## 1️⃣ Clone the repository

```bash
git clone https://github.com/Abin-Binu123/Evoastra_Major_Project.git
cd Evoastra_Major_Project
```

## 2️⃣ Create a virtual environment

```bash
python -m venv venv
```

## 3️⃣ Activate venv

Windows:

```bash
venv\Scripts\activate
```

## 4️⃣ Install dependencies

```bash
pip install fastapi uvicorn pillow torch torchvision transformers nltk python-multipart
```

## 5️⃣ Download NLTK tokenizer

```python
import nltk
nltk.download('punkt')
```

---

# Running the Backend

```bash
uvicorn backend.caption_api:app --port 8080
```

You should see:

```
Uvicorn running on http://127.0.0.1:8080
```

Test backend:
👉 [http://127.0.0.1:8080/](http://127.0.0.1:8080/)

---

Running the Frontend

Just **double-click `index.html`** and open it in the browser.

Features:

* Auto-detect backend on port **8000** or **8080**
* Upload image
* See preview
* See caption + BLEU score
* Clean UI with spinner

---

Example Output

```
Caption: "A dog running across a grassy field"
BLEU Score: 0.91
```

Future Scope

* Multi-language captioning
* Faster BLIP model (BLIP-2/Flan-T5 integration)
* Cloud deployment (Render, HuggingFace Spaces)
* Mobile app integration
* Voice-based caption reading




