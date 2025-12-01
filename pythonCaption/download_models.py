from transformers import BlipProcessor, BlipForConditionalGeneration

model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

#Save to local directory
import os
os.makedirs("./models/blip", exist_ok=True)
processor.save_pretrained("./models/blip")
model.save_pretrained("./models/blip")