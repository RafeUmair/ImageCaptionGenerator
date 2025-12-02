from transformers import BlipProcessor, BlipForConditionalGeneration
import os;

#Download BLIP model
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

#Save Blip model locally
os.makedirs("./models/blip", exist_ok=True)
processor.save_pretrained("./models/blip")
model.save_pretrained("./models/blip")

