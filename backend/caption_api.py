from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import io
import math

# ---------------------------
# INITIAL SETUP
# ---------------------------

nltk.download('punkt', quiet=True)

app = FastAPI(title="Evoastra Image Captioning API 🚀")

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# LOAD BLIP MODEL
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    cache_dir="E:/hf_cache"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    cache_dir="E:/hf_cache"
).to(device)

print("✅ Pretrained BLIP model loaded successfully!")


# ---------------------------
# CAPTION + BLEU FUNCTION
# ---------------------------

def generate_caption_with_bleu(image: Image.Image):
    # -------- Generate Caption A (greedy decoding) --------
    inputs = processor(images=image, return_tensors="pt").to(device)
    output_ids_1 = model.generate(**inputs, max_new_tokens=20)
    caption_1 = processor.decode(output_ids_1[0], skip_special_tokens=True).strip()

    # -------- Generate Caption B (sampling decoding) --------
    output_ids_2 = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,        # enable randomness
        top_k=50,              # top-k sampling
        top_p=0.95             # nucleus sampling
    )
    caption_2 = processor.decode(output_ids_2[0], skip_special_tokens=True).strip()

    # Tokenize captions
    tokens_1 = nltk.word_tokenize(caption_1)
    tokens_2 = nltk.word_tokenize(caption_2)

    # Compute BLEU between Caption A and Caption B
    try:
        bleu = sentence_bleu(
            [tokens_1], 
            tokens_2,
            smoothing_function=SmoothingFunction().method1
        )
    except:
        bleu = 0.0

    # Avoid NaN
    if math.isnan(bleu):
        bleu = 0.0

    return caption_1, round(float(bleu), 3)



# ---------------------------
# ROUTES
# ---------------------------

@app.get("/")
def home():
    return {"message": "Evoastra Captioning API with BLIP + BLEU is running 🚀"}


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        # Read uploaded file bytes
        contents = await file.read()

        # Convert to PIL image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Generate caption + BLEU
        caption, bleu = generate_caption_with_bleu(image)

        # Safety checks
        if not caption:
            caption = "No caption generated"
        if bleu is None or bleu < 0:
            bleu = 0.0

        print(f"🖼️ {file.filename} → {caption} (BLEU: {bleu})")

        return JSONResponse({"caption": caption, "bleu": bleu})

    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse(
            {"caption": "Error generating caption", "bleu": 0.0, "error": str(e)},
            status_code=500
        )


# Run using:
# uvicorn backend.caption_api:app --port 8080
