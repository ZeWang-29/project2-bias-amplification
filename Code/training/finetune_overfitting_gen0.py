"""
Fine-tune base GPT-2 with the Overfitting mitigation strategy to produce Generation 0.

Differences from the standard setup:
  - 25 epochs (5x baseline)
  - weight_decay = 0 (no regularization)

Paper reference: Section 4.3 (Mitigation Strategies — Overfitting)
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# ============================================================
# Configuration — adjust these paths to your environment
# ============================================================
REAL_DATA_PATH = "D_mixed.txt"          # 1,518 real articles
OUTPUT_MODEL_DIR = "models/MMO1"        # Output directory for Overfitting Generation 0 model

# ============================================================
# Fine-tuning (Overfitting setup: 25 epochs, weight_decay=0)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("text", data_files={"train": REAL_DATA_PATH})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    overwrite_output_dir=True,
    num_train_epochs=25,                # 5x baseline to encourage overfitting
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=200,
    learning_rate=5e-5,
    weight_decay=0.0,                   # No regularization
    evaluation_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print(f"Overfitting Generation 0 model saved to {OUTPUT_MODEL_DIR}")
