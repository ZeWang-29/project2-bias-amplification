"""
Fine-tune base GPT-2 on the real mixed dataset to produce the Generation 0 model.

This script fine-tunes GPT-2 on D_mixed.txt (1,518 real news articles: 506 Left + 506 Center + 506 Right)
to produce the initial model (Generation 0) for the iterative synthetic training pipeline.

Paper reference: Section 3.2 (Successive Fine-tuning)
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# ============================================================
# Configuration — adjust these paths to your environment
# ============================================================
REAL_DATA_PATH = "D_mixed.txt"          # 1,518 real articles (506 Left + 506 Center + 506 Right)
OUTPUT_MODEL_DIR = "models/MM1"         # Output directory for Generation 0 model

# ============================================================
# Fine-tuning
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
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=200,
    learning_rate=5e-5,
    weight_decay=0.01,
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
print(f"Generation 0 model saved to {OUTPUT_MODEL_DIR}")
