from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

#Load BLIP model once at startup
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

app = FastAPI(title="Local Image Caption Generator")

@app.post("/caption")
async def generate_caption(image: UploadFile = File(...)):  
    try:
        #Read uploaded image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        #Generate caption
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
       
        #Capitalize first letter of caption
        if caption:
            caption = caption[0].upper() + caption[1:]
            return JSONResponse(content=[caption])

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
