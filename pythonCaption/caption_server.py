from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from googletrans import Translator  
import io

model_path = "/app/models/blip"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
translator = Translator()


app = FastAPI(title="Local Image Caption Generator")

@app.post("/caption")
async def generate_caption(image: UploadFile = File(...), 
                           language: str = Form("en"),
                            tone: str = Form("normal")):  
    try:
        #Read uploaded image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        #Generate caption
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        #Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]

        # Tone adjustment - placeholder for future implementation
        # Currently just returns the original caption regardless of tone
        if tone != "normal":
            # TODO: Implement tone adjustment with your new method
            pass

        #Translate if language is not English
        if language != "en":
            translated = translator.translate(caption, dest=language)
            caption = translated.text

        return JSONResponse(content=[caption])

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
