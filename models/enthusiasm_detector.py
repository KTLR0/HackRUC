# enthusiasm_detector.py

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os

# -------------------------------
# Step 0: Parse arguments
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--save_model_path', type=str, default='saved_models/enthusiasm_model.pt',
                    help='Path to save the trained model')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
args = parser.parse_args()

# Ensure save folder exists
os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)

# -------------------------------
# Step 1: Load and preprocess dataset
# -------------------------------
print("Loading GoEmotions dataset...")
dataset = load_dataset("go_emotions")

# Map GoEmotions labels to enthusiastic (1) or not (0)
enthusiastic_labels = [3, 7, 11]  # joy, excitement, love (example)

def map_labels(emo_list):
    return 1 if any(label in enthusiastic_labels for label in emo_list) else 0

labels = [map_labels(l) for l in dataset['train']['labels']]
texts = dataset['train']['text']

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# -------------------------------
# Step 2: Tokenization
# -------------------------------
print("Tokenizing...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(texts, max_length=128):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

train_encodings = tokenize(X_train)
val_encodings = tokenize(X_val)

# -------------------------------
# Step 3: PyTorch Dataset
# -------------------------------
class EnthusiasmDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EnthusiasmDataset(train_encodings, y_train)
val_dataset = EnthusiasmDataset(val_encodings, y_val)

# -------------------------------
# Step 4: Load BERT model
# -------------------------------
print("Loading BERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# -------------------------------
# Step 5: Training components
# -------------------------------
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# -------------------------------
# Step 6: Training loop
# -------------------------------
for epoch in range(args.num_epochs):
    # Training
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1} | Validation Accuracy: {val_acc:.4f}\n")

# -------------------------------
# Step 7: Save trained model
# -------------------------------
torch.save(model.state_dict(), args.save_model_path)
print(f"Model saved to {args.save_model_path}")

# -------------------------------
# Step 8: Prediction function
# -------------------------------
def predict(text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        return "Enthusiastic" if pred == 1 else "Not Enthusiastic"

# -------------------------------
# Step 9: Test examples
# -------------------------------
if __name__ == "__main__":
    test_texts = [
        "I absolutely love this project! It's amazing!!!",
        "I feel okay about this.",
        "This is so exciting, I can't wait!",
        "I am bored and uninterested."
    ]
    for t in test_texts:
        print(f"'{t}' -> {predict(t, model, tokenizer, device)}")
