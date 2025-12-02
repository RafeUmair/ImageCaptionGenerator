from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer
from PIL import Image
from googletrans import Translator  
import io
import torch

#Load BLIP model for image captioning
model_path = "/app/models/blip"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

#Load custom trained T5 model for tone modification
custom_model_path = "/app/models/t5-caption-style"
tone_tokenizer = T5Tokenizer.from_pretrained(custom_model_path)
tone_model = T5ForConditionalGeneration.from_pretrained(custom_model_path)

#Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tone_model.to(device)
print(f"Custom trained model loaded on {device}")

translator = Translator()
app = FastAPI(title="Local Image Caption Generator with Custom Model")

def modify_tone(caption, tone):
    if tone == "normal":
        return caption
    
    try:

        input_text = f"Make {tone}: {caption}"
        
        inputs = tone_tokenizer(
            input_text,
            return_tensors='pt',
            max_length=128,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        #Generate with custom model
        with torch.no_grad():
            outputs = tone_model.generate(
                **inputs,
                max_length=100 if tone != 'short' else 20,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        modified_caption = tone_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        #Validate and clean output
        if modified_caption and len(modified_caption.strip()) > 3:
            modified_caption = modified_caption.strip()
            
            
            if modified_caption:
                modified_caption = modified_caption[0].upper() + modified_caption[1:]
            
            print(f"✓ Original: {caption}")
            print(f"✓ Tone: {tone}")
            print(f"✓ Modified: {modified_caption}")
            
            return modified_caption
        
        #Fallback to original if output is invalid
        print(f"Model output invalid, returning original")
        return caption
        
    except Exception as e:
        print(f"✗ Tone modification error: {e}")
        return caption

@app.post("/caption")
async def generate_caption(image: UploadFile = File(...), 
                           language: str = Form("en"),
                           tone: str = Form("normal")):  
    try:
        #Read image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        #Generate caption with BLIP
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        if caption:
            caption = caption[0].upper() + caption[1:]

        #Apply tone modification
        if tone != "normal":
            caption = modify_tone(caption, tone)

        #Translate language if needed
        if language != "en":
            translated = translator.translate(caption, dest=language)
            caption = translated.text

        return JSONResponse(content=[caption])

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})