"""
Fine-tune Generation 0 model with the Accumulation strategy to produce Generation 1.

Accumulation: trains on ALL previous datasets combined (real data D0 + synthetic data D1).

Paper reference: Section 4.3 (Mitigation Strategies — Accumulation)
"""

import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Configuration — adjust these paths to your environment
# ============================================================
DATASET_PATHS = [
    "D_mixed.txt",                  # Original real data (D0)
    "synthetic_data/DDA1.txt",      # Synthetic data from Generation 0
]
INITIAL_MODEL = "models/MM1"       # Generation 0 model (or "refipsai/MM1")
OUTPUT_MODEL_DIR = "models/MMA2"   # Output: Accumulation Generation 1 model

# ============================================================
# Helper functions
# ============================================================
def read_articles_from_text(file_path):
    articles = []
    current_article = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip().startswith('"title:'):
                if current_article:
                    article_body = "".join(current_article).strip()
                    articles.append({"body": article_body, "formatted": article_body})
                    current_article = []
            current_article.append(line)
        if current_article:
            article_body = "".join(current_article).strip()
            articles.append({"body": article_body, "formatted": article_body})
    return pd.DataFrame(articles)


# ============================================================
# Combine all datasets and fine-tune
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(INITIAL_MODEL)
model = AutoModelForCausalLM.from_pretrained(INITIAL_MODEL)
model.to(device)

dataframes = [read_articles_from_text(path) for path in DATASET_PATHS]
combined_data = pd.concat(dataframes, ignore_index=True)

combined_file_path = "combined_training_data_MMA2.txt"
with open(combined_file_path, "w") as f:
    for article in combined_data["formatted"]:
        f.write(f"{article}\n\n")

dataset = load_dataset("text", data_files={"train": combined_file_path})
tokenizer.pad_token = tokenizer.eos_token

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
print(f"Accumulation Generation 1 model saved to {OUTPUT_MODEL_DIR}")
