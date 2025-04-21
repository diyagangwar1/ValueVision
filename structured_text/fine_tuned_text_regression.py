# fine_tune_text_regression.py

"""
Fine-tuning DistilBERT for Real Estate Price Prediction from Text Descriptions
------------------------------------------------------------------------------

This script fine-tunes a pretrained DistilBERT transformer to predict house prices
based solely on the natural language descriptions found in housing listings. It uses
a regression head (num_labels=1) on top of the model and is trained using Hugging Face's
Trainer API. The final model captures domain-specific semantic features relevant to home
valuation from listing text, and serves as the text-only baseline in a larger multi-modal
project that will later integrate structured and image data.

Key Features:
- Merges description and price data via property ID
- Tokenizes text for BERT-compatible input
- Fine-tunes BERT for regression using train/validation split
- Outputs a model that can predict prices from listing descriptions alone

"""


import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Description Data
desc_df = pd.read_csv("raw_data/description_data.csv") 
price_df = pd.read_csv("raw_data/property.csv") 
price_df['price'] = pd.to_numeric(price_df['price'], errors='coerce')
price_df = price_df.dropna(subset=['price'])

# outlier removal (top 1%)
price_threshold = price_df['price'].quantile(0.99)
price_df = price_df[price_df['price'] <= price_threshold]
df = desc_df.merge(price_df[['ID', 'price']], on='ID', how='inner')
df['text'] = df['des_head'].fillna('') + ' ' + df['des_content'].fillna('')
df = df.dropna(subset=['text'])
texts = df['text'].tolist()
prices = df['price'].astype(float).tolist()

# Normalize price labels
scaler = StandardScaler()
prices_scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1)).flatten()

# Split into Train/Validation Sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, prices_scaled, test_size=0.2, random_state=42
)

# Dataset and Tokenization
# loads a pre-trained BERT tokenizer that converts raw text into token
# IDS and attention masks for model input
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class HousingTextDataset(Dataset):
    # custom PyTorch dataset class
    # prepares data for Hugging Face Trainer to iterate over batches during training
    def __init__(self, texts, labels, tokenizer):
        # When the dataset is initialized, tokenize all the text in advance
        self.encodings = tokenizer(
            texts,                   # list of input texts
            truncation=True,         # cut off long texts at 512 tokens
            padding=True,            # pad short texts to the same length
            max_length=256
        )
        self.labels = labels         # numeric prices to be predicted

    def __getitem__(self, idx):
        # Return one training example at a time (indexed by idx)
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # This gives us: item['input_ids'], item['attention_mask']
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.labels)

# Instantiate train and validation datasets
train_dataset = HousingTextDataset(train_texts, train_labels, tokenizer)
val_dataset = HousingTextDataset(val_texts, val_labels, tokenizer)

# Load Model with Regression Head
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    #"bert-base-uncased",
    num_labels=1  # regression
)

batch_size = 2  # Same as per_device_catrain_batch_size
steps_per_epoch = len(train_dataset) // batch_size

# Define Training Arguments 
training_args = TrainingArguments(
    output_dir='./results_text_only',         # Folder to save model checkpoints and final model
    num_train_epochs=2,                       # Total number of passes through the entire training dataset
    per_device_train_batch_size=batch_size,            # Number of samples per GPU/CPU batch during training
    per_device_eval_batch_size=batch_size,             # Number of samples per batch during evaluation
    learning_rate=3e-5,                       # Initial learning rate for AdamW optimizer (standard for BERT)
    weight_decay=0.01,                        # L2 regularization strength to help prevent overfitting
    #evaluation_strategy="no",                # Evaluate on the validation set once per epoch
    #save_strategy="epoch",                   # Save the model after each epoch 
    max_grad_norm=1.0,
    save_steps=steps_per_epoch,
    logging_dir='./logs_text_only',           # Directory to store logs for TensorBoard visualization
    logging_steps=20,                         # Log training metrics every 20 steps (batches)
    report_to="none"
    #evaluate_during_training=True
)

# Define custom metrics: MAE and RMSE
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.squeeze()
    mae = mean_absolute_error(labels, preds)
    rmse = mean_squared_error(labels, preds, squared=False)
    return {
        "MAE": mae,
        "RMSE": rmse,
    }

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# begins the fine-tining process, where BERT's parameters are updated
trainer.train()

# Save Model
model.save_pretrained('./fine_tuned_text_only_model')
tokenizer.save_pretrained('./fine_tuned_text_only_model')


# Predict price from new listing text
def predict_price(description, model, tokenizer):
    # Generates a predicted price for one sample description
    model.eval()
    inputs = tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
        scaled_pred = output.logits.item()
    return scaler.inverse_transform([[scaled_pred]])[0][0]

# Example usage
example = "Spacious 4-bedroom family home with modern kitchen and large backyard."
price = predict_price(example, model, tokenizer)
print(f"\nPredicted price for sample listing: ${price:,.2f}")


# Plot Training and Validation Losses
# Extract loss values from training logs
training_logs = trainer.state.log_history
train_loss = [log['loss'] for log in training_logs if 'loss' in log]
eval_loss = [log['eval_loss'] for log in training_logs if 'eval_loss' in log]
epochs = list(range(1, len(eval_loss) + 1))

# Plot the learning curves
plt.figure(figsize=(8, 5))
plt.plot(epochs, eval_loss, label='Validation Loss', marker='o')
plt.plot(epochs, train_loss[:len(epochs)], label='Training Loss (approx)', linestyle='--', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve: Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve.png")
plt.show()

# Save Predictions to CSV

# Run predictions on validation set
predictions = trainer.predict(val_dataset)
predicted_prices = predictions.predictions.squeeze()
true_prices = predictions.label_ids

results_df = pd.DataFrame({
    'description': val_texts,
    'actual_price': true_prices,
    'predicted_price': predicted_prices
})
results_df.to_csv("predictions.csv", index=False)
print("\n Predictions saved to predictions.csv")
