import torch
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd

# Load model and tokenizer from your checkpoint

checkpoint_path = "/Users/diyagangwar/cs158/project/results_text_only/checkpoint-42576"
model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)

training_args = TrainingArguments(output_dir="./results_text_only", report_to="none")
trainer = Trainer(model=model, args=training_args)

# Load trainer state manually
import json

with open(f"{checkpoint_path}/trainer_state.json", "r") as f:
    trainer_state = json.load(f)

logs = trainer_state["log_history"]

# Extract losses
train_logs = [entry for entry in logs if 'loss' in entry]

# Extract steps and losses
steps = [entry['step'] for entry in train_logs]
losses = [entry['loss'] for entry in train_logs]

window_size = 50
losses_smooth = pd.Series(losses).rolling(window=window_size).mean()

plt.figure(figsize=(10, 6))
plt.plot(steps, losses_smooth, label="Training Loss (Smoothed)", color="blue")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Smoothed Training Loss Curve (DistilBERT Fine-Tuning)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("smoothed_training_loss_curve.png", dpi=300)
plt.show()
