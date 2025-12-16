import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import random
import os

def generate_training_data():
    base_captions = [
        "a dog sitting on grass", "a cat sleeping on a bed", "a person holding a cup of coffee",
        "a bird flying in the sky", "a car parked on the street", "children playing in a park",
        "a sunset over mountains", "a flower in a garden", "a laptop on a desk",
        "people walking on a beach", "a tree in the forest", "a building in the city",
        "food on a plate", "a phone on a table", "books on a shelf", "a painting on a wall",
        "clouds in the sky", "a bicycle on a path", "a window with curtains",
        "a door with a handle", "a bridge over water", "a cat playing with a toy",
        "a dog running in a field", "a person reading a book", "a cup on a saucer",
        "a chair at a table", "a lamp on a nightstand", "a computer monitor on a desk",
        "a plant in a pot", "a clock on the wall", "a picture frame on a shelf",
        "a candle burning brightly", "a bag on the floor", "a hat on a hook",
        "a pair of shoes by the door", "a mirror reflecting light", "a vase with flowers",
        "a bowl of fruit", "a glass of water", "a plate with food", "a spoon on the table",
        "a woman standing next to a tree", "a man riding a bicycle down a street",
        "a group of people sitting at a table", "a cat looking out a window",
        "a dog running through a field", "a bird perched on a branch",
        "a child playing with a toy", "a person typing on a laptop",
        "a couple walking along the beach", "a city skyline at night",
        "a mountain covered in snow", "a river flowing through a valley",
        "a red car on the road", "a white house with a garden",
        "a black cat on a couch", "a blue sky with white clouds",
        "a green tree in the park", "a yellow flower blooming",
        "a person sitting on a bench", "a man walking down the street",
        "a woman holding a bag", "a child riding a bike",
        "a dog playing in water", "a cat sitting by the window",
        "a sunset over the ocean", "a sunrise in the morning",
        "a full moon at night", "a cup of tea on a table",
        "a glass of juice", "a bottle of water on a desk",
        "a sandwich on a plate", "a pizza on the table",
        "a bowl of soup", "a person wearing a hat",
        "a man in a suit", "a woman in a dress",
        "a dog with a ball", "a cat with a collar",
        "a car in a parking lot", "a truck on the highway",
        "a train on the tracks", "a boat on the water",
        "a house with a garden", "a building with windows",
        "a street with cars", "a road through the forest",
        "a path in the park", "a table with chairs",
        "a desk with a computer", "a shelf with books",
        "a wall with pictures", "a room with furniture",
        "a kitchen with appliances", "a bedroom with a bed",
        "a garden with flowers", "a yard with grass",
    ]
    
    funny_templates = [
        "plot twist: {caption}, absolutely slaying the game",
        "breaking: {caption}, caught being ridiculously perfect",
        "pov: you're witnessing {caption} and it's everything",
        "ladies and gentlemen, {caption}, and it's iconic",
        "nobody asked but here's {caption} living its best life",
        "when you see {caption} just existing flawlessly",
        "{caption} - 10/10, would recommend, no notes",
        "the council has unanimously agreed that {caption} is art",
        "sir/madam, this is {caption} and we're not ready",
        "behold! {caption}, serving looks since forever",
        "peak performance: {caption}",
        "{caption} hitting different today",
        "iconic moment: {caption} being legendary",
        "{caption} said 'watch this' and delivered",
        "the vibes: {caption} immaculate",
        "{caption} living the dream honestly",
        "when {caption} is just *chef's kiss*",
        "{caption} pure art no notes",
    ]
    
    poetic_templates = [
        "behold, {caption}, captured in time's gentle embrace",
        "here dwells {caption}, a moment of ethereal beauty",
        "in this frame, {caption}, where stillness meets grace",
        "upon this canvas, {caption}, painted by light divine",
        "witness {caption}, a symphony frozen in eternal time",
        "through the lens, {caption}, forever preserved in beauty",
        "where {caption} rests, serenity whispers its ancient tale",
        "in quiet repose, {caption}, dwelling in peaceful harmony",
        "soft and serene, {caption}, bathed in gentle radiance",
        "{caption}, a vision of tranquil magnificence",
        "a moment preserved: {caption}",
        "where {caption}, time stands still",
        "{caption}, bathed in gentle light",
        "softly rests {caption}, peaceful and serene",
        "here lies {caption}, quiet and beautiful",
        "{caption}, frozen in time's embrace",
        "witness {caption}, dwelling in tranquil grace",
        "in stillness, {caption}, beauty unfolds",
    ]
    
    formal_templates = [
        "the photographic documentation depicts {caption}, as recorded in the visual archive",
        "this image presents {caption}, captured through standardized imaging methodology",
        "visual analysis reveals {caption}, preserved for observational documentation",
        "the composition illustrates {caption}, demonstrating key structural elements",
        "documentation indicates {caption}, recorded in high-resolution format",
        "the captured frame presents {caption}, as evidenced in the visual record",
        "observational data shows {caption}, documented for research purposes",
        "the visual medium depicts {caption}, catalogued in the archival system",
        "this image depicts {caption}",
        "the photograph shows {caption}",
        "visible in this frame is {caption}",
        "the composition features {caption}",
        "documented: {caption}",
        "captured here is {caption}",
        "the scene displays {caption}",
        "illustrated in this photograph: {caption}",
    ]
    
    training_data = []
    
    #Generate training pairs
    for caption in base_captions:
        #Add funny variations
        for temp in funny_templates:
            training_data.append({
                "input": f"Make funny: {caption}",
                "output": temp.format(caption=caption.lower())
            })

        #Add poetic variations
        for temp in poetic_templates:
            training_data.append({
                "input": f"Make poetic: {caption}",
                "output": temp.format(caption=caption.lower())
            })

        #Add formal variations
        for temp in formal_templates:
            training_data.append({
                "input": f"Make formal: {caption}",
                "output": temp.format(caption=caption.lower())
            })

        #Short version 
        short = ' '.join(
            caption.lower().replace('a ', '').replace('an ', '').replace('the ', '').split()[:4]
        ).capitalize()

        training_data.append({
            "input": f"Make short: {caption}",
            "output": short
        })

    random.shuffle(training_data)
    print(f"Generated {len(training_data)} training examples")
    return training_data


class CaptionStyleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_enc = self.tokenizer(
            item["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        output_enc = self.tokenizer(
            item["output"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = output_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }


def train_model(epochs=10, batch_size=8, learning_rate=5e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training on: {device}")
    
    #Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    
    print("Generating training data...")
    training_data = generate_training_data()
    
    #Save sample for inspection
    with open("training_data_sample.json", "w") as f:
        json.dump(training_data[:50], f, indent=2)
    
    #Create dataset and dataloader
    dataset = CaptionStyleDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    #Optimizer with weight decay for regularization
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler for better convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    print(f"Training for {epochs} epochs...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Total batches per epoch: {len(dataloader)}")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )
            
            loss = outputs.loss
            loss.backward()
            
            #gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            #Print progress every 100 batches
            if batch_count % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_count}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        #Calculate average loss
        avg_loss = total_loss / batch_count
        
        #Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{epochs} completed")

        
        #Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("./models/t5-caption-style-best", exist_ok=True)
            model.save_pretrained("./models/t5-caption-style-best")
            tokenizer.save_pretrained("./models/t5-caption-style-best")
    
    #Save final model
    os.makedirs("./models/t5-caption-style", exist_ok=True)
    model.save_pretrained("./models/t5-caption-style")
    tokenizer.save_pretrained("./models/t5-caption-style")
    
    print(f"Training complete")



if __name__ == "__main__":
    train_model(epochs=10, batch_size=8, learning_rate=5e-4)
    print("Model training finished")