import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from tqdm import tqdm
import json
import random
import os

#Generate Training Data
def generate_training_data():
    
    #Base captions
    base_captions = [
        "A dog sitting on grass",
        "A cat sleeping on a bed",
        "A person holding a cup of coffee",
        "A bird flying in the sky",
        "A car parked on the street",
        "Children playing in a park",
        "A sunset over mountains",
        "A flower in a garden",
        "A laptop on a desk",
        "People walking on a beach",
        "A tree in the forest",
        "A building in the city",
        "Food on a plate",
        "A phone on a table",
        "Books on a shelf",
        "A painting on a wall",
        "Clouds in the sky",
        "A bicycle on a path",
        "A window with curtains",
        "A door with a handle",
        "A bridge over water",
        "A cat playing with a toy",
        "A dog running in a field",
        "A person reading a book",
        "A cup on a saucer",
        "A chair at a table",
        "A lamp on a nightstand",
        "A computer monitor on a desk",
        "A plant in a pot",
        "A clock on the wall",
        "A picture frame on a shelf",
        "A candle burning brightly",
        "A bag on the floor",
        "A hat on a hook",
        "A pair of shoes by the door",
        "A mirror reflecting light",
        "A vase with flowers",
        "A bowl of fruit",
        "A glass of water",
        "A plate with food",
        "A spoon on the table",
        # ADD 150+ MORE for best results
    ]
    
    training_data = []
    
    #FUNNY tone templates
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
    
    #POETIC tone templates
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
    
    #FORMAL tone templates
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
    
    print(f"Generating training data from {len(base_captions)} base captions...")
    
    #Generate examples for each tone
    for caption in base_captions:
        #Funny
        for template in funny_templates:
            training_data.append({
                "input": f"Make funny: {caption}",
                "output": template.format(caption=caption.lower())
            })
        
        #Poetic
        for template in poetic_templates:
            training_data.append({
                "input": f"Make poetic: {caption}",
                "output": template.format(caption=caption.lower())
            })
        
        #Formal
        for template in formal_templates:
            training_data.append({
                "input": f"Make formal: {caption}",
                "output": template.format(caption=caption.lower())
            })
        
        #Short
        words = caption.lower().replace('a ', '').replace('an ', '').replace('the ', '').split()
        short = ' '.join(words[:4]).capitalize()
        training_data.append({
            "input": f"Make short: {caption}",
            "output": short
        })
    
    #Shuffle for better training
    random.shuffle(training_data)
    
    print(f" Generated {len(training_data)} training examples")
    print(f"  - Funny: {len(base_captions) * len(funny_templates)}")
    print(f"  - Poetic: {len(base_captions) * len(poetic_templates)}")
    print(f"  - Formal: {len(base_captions) * len(formal_templates)}")
    print(f"  - Short: {len(base_captions)}")
    
    return training_data


#Dataset Class
class CaptionStyleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        #Tokenize input
        input_encoding = self.tokenizer(
            item['input'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        #Tokenize output
        target_encoding = self.tokenizer(
            item['output'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        #Prepare labels
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


#Training Function
def train_model(epochs=3, batch_size=4, learning_rate=3e-4):

    #Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("Training on CPU...\n")
    else:
        print(f"GPU detected. Training on GPU\n")
    
    #Load pre-trained T5
    print("Loading T5-small model..")
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    print("Model loaded\n")
    
    #Generate training data
    print("Generating training data...")
    training_data = generate_training_data()
    
    #Save training data
    with open('training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    print("✓ Training data saved to training_data.json\n")
    
    #Create dataset and dataloader
    print("Creating dataset...")
    dataset = CaptionStyleDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset ready: {len(dataset)} examples, {len(dataloader)} batches\n")
    
    #Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    #Training loop
    print("Starting training...\n")
    model.train()
    
    for epoch in range(epochs):
        print(f"{'='*60}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            #Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            #Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        print(f"\n✓ Epoch {epoch+1} complete - Average Loss: {avg_loss:.4f}\n")
    
    #Save the fine tuned model
    output_dir = "./models/t5-caption-style"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"✓ Model saved to: {output_dir}")
    print(f"✓ Training data: training_data.json")    
    return model, tokenizer


#Main execution
if __name__ == "__main__":
    import sys
     
    #Train the model
    print("\nStarting training...\n")
    train_model(epochs=3, batch_size=4, learning_rate=3e-4)
    
    print("\n Model training finished")