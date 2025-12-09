import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
import json
import random
import os

#Generate Training Data
def generate_training_data():
    base_captions = [
        "A dog sitting on grass", "A cat sleeping on a bed", "A person holding a cup of coffee",
        "A bird flying in the sky", "A car parked on the street", "Children playing in a park",
        "A sunset over mountains", "A flower in a garden", "A laptop on a desk",
        "People walking on a beach", "A tree in the forest", "A building in the city",
        "Food on a plate", "A phone on a table", "Books on a shelf", "A painting on a wall",
        "Clouds in the sky", "A bicycle on a path", "A window with curtains",
        "A door with a handle", "A bridge over water", "A cat playing with a toy",
        "A dog running in a field", "A person reading a book", "A cup on a saucer",
        "A chair at a table", "A lamp on a nightstand", "A computer monitor on a desk",
        "A plant in a pot", "A clock on the wall", "A picture frame on a shelf",
        "A candle burning brightly", "A bag on the floor", "A hat on a hook",
        "A pair of shoes by the door", "A mirror reflecting light", "A vase with flowers",
        "A bowl of fruit", "A glass of water", "A plate with food", "A spoon on the table",
    ]
    
    funny_templates = [
        "Plot twist: {caption}, absolutely slaying the game",
        "Breaking: {caption}, caught being ridiculously perfect",
        "POV: You're witnessing {caption} and it's everything",
        "Ladies and gentlemen, {caption}, and it's iconic",
        "Nobody asked but here's {caption} living its best life",
        "When you see {caption} just existing flawlessly",
        "{caption} - 10/10, would recommend, no notes",
        "The council has unanimously agreed that {caption} is art",
        "Sir/Madam, this is {caption} and we're not ready",
        "Behold! {caption}, serving looks since forever",
    ]
    
    poetic_templates = [
        "Behold, {caption}, captured in time's gentle embrace",
        "Here dwells {caption}, a moment of ethereal beauty",
        "In this frame, {caption}, where stillness meets grace",
        "Upon this canvas, {caption}, painted by light divine",
        "Witness {caption}, a symphony frozen in eternal time",
        "Through the lens, {caption}, forever preserved in beauty",
        "Where {caption} rests, serenity whispers its ancient tale",
        "In quiet repose, {caption}, dwelling in peaceful harmony",
        "Soft and serene, {caption}, bathed in gentle radiance",
        "{caption}, a vision of tranquil magnificence",
    ]
    
    formal_templates = [
        "The photographic documentation depicts {caption}, as recorded in the visual archive",
        "This image presents {caption}, captured through standardized imaging methodology",
        "Visual analysis reveals {caption}, preserved for observational documentation",
        "The composition illustrates {caption}, demonstrating key structural elements",
        "Documentation indicates {caption}, recorded in high-resolution format",
        "The captured frame presents {caption}, as evidenced in the visual record",
        "Observational data shows {caption}, documented for research purposes",
        "The visual medium depicts {caption}, catalogued in the archival system",
    ]
    
    training_data = []
    
    for caption in base_captions:
        for temp in funny_templates:
            training_data.append({"input": f"Make funny: {caption}",
                                  "output": temp.format(caption=caption.lower())})

        for temp in poetic_templates:
            training_data.append({"input": f"Make poetic: {caption}",
                                  "output": temp.format(caption=caption.lower())})

        for temp in formal_templates:
            training_data.append({"input": f"Make formal: {caption}",
                                  "output": temp.format(caption=caption.lower())})

        short = ' '.join(
            caption.lower().replace('a ','').replace('an ','').replace('the ','').split()[:4]
        ).capitalize()
        
        training_data.append({
            "input": f"Make short: {caption}",
            "output": short
        })
    
    random.shuffle(training_data)
    return training_data


#Dataset
class CaptionStyleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_enc = self.tokenizer(item["input"], max_length=self.max_length,
                                   padding="max_length", truncation=True, return_tensors="pt")
        output_enc = self.tokenizer(item["output"], max_length=self.max_length,
                                    padding="max_length", truncation=True, return_tensors="pt")
        
        labels = output_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }


#Training Function
def train_model(epochs=3, batch_size=4, learning_rate=3e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

    print("Generating training data...")
    training_data = generate_training_data()

    with open("training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    dataset = CaptionStyleDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print("Training started...")

    model.train()
    for _ in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    os.makedirs("./models/t5-caption-style", exist_ok=True)
    model.save_pretrained("./models/t5-caption-style")
    tokenizer.save_pretrained("./models/t5-caption-style")

    print("Training complete.")


# Main
if __name__ == "__main__":
    train_model(epochs=3, batch_size=4, learning_rate=3e-4)
    print("Model training finished")
