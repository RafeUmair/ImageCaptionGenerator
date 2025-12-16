from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer
from PIL import Image
from googletrans import Translator  
import io
import torch

#Load BLIP model
model_path = "/app/models/blip"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

#Load custom T5 model
custom_model_path = "/app/models/t5-caption-style"
tone_tokenizer = T5Tokenizer.from_pretrained(custom_model_path)
tone_model = T5ForConditionalGeneration.from_pretrained(custom_model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tone_model.to(device)
model.to(device)
print(f" Models loaded on {device}")

translator = Translator()
app = FastAPI(title="Image Caption Generator")

#Tone Modification
def modify_tone(caption, tone):
    if tone == "normal":
        return caption

    original_caption = caption
    caption_lower = caption.lower()

    #Format input for T5
    input_text = f"Make {tone}: {caption_lower}"
    inputs = tone_tokenizer(
        input_text,
        return_tensors='pt',
        max_length=128,
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    #Generate modified caption
    with torch.no_grad():
        outputs = tone_model.generate(
            **inputs,
            max_length=100 if tone != 'short' else 20,
            min_length=len(caption_lower.split()) if tone != 'short' else 2,
            num_beams=6,
            temperature=0.7,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=0.9,
        )

    modified_caption = tone_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    #Validation
    #Minimum length
    if len(modified_caption) < 5 and tone != 'short':
        print(f" Output too short, using original")
        return original_caption

    #Word overlap
    original_words = set(caption_lower.split())
    modified_words = set(modified_caption.lower().split())
    overlap_ratio = len(original_words & modified_words) / len(original_words) if original_words else 0
    if overlap_ratio < 0.4 and tone != 'short':
        print(f" Low overlap ({overlap_ratio:.1%}), possible hallucination")
        return original_caption

    #Output same as input
    if modified_caption.lower() == caption_lower:
        print(f" Output same as input, using original")
        return original_caption

    #Capitalize
    if modified_caption and modified_caption[0].islower():
        modified_caption = modified_caption[0].upper() + modified_caption[1:]

    print(f" Original: {caption}")
    print(f" Tone: {tone}")
    print(f" Modified: {modified_caption}")

    return modified_caption

#Caption Endpoint
@app.post("/caption")
async def generate_caption(
    image: UploadFile = File(...),
    language: str = Form("en"),
    tone: str = Form("normal")
):
    try:
        #Read image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        #Generate BLIP caption
        inputs = processor(images=img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        #Capitalize
        if caption and caption[0].islower():
            caption = caption[0].upper() + caption[1:]

        print(f"BLIP caption: {caption}")

        #Aply tone
        if tone in ["funny", "poetic", "formal", "short"]:
            caption = modify_tone(caption, tone)
        elif tone != "normal":
            print(f" Invalid tone '{tone}', using normal")

        #Translate if needed
        if language != "en":
            try:
                translated = translator.translate(caption, dest=language)
                caption = translated.text
                print(f"Translated to {language}")
            except Exception as e:
                print(f"Translation error: {e}")

        return JSONResponse(content=[caption])

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
