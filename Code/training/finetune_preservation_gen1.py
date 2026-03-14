"""
Fine-tune Generation 0 model with the Preservation strategy to produce Generation 1.

Preservation: adds 10% (152) randomly selected real articles to the synthetic training data.

Paper reference: Section 4.3 (Mitigation Strategies — Preservation)
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
SYNTHETIC_DATA_PATH = "synthetic_data/DD1.txt"  # Synthetic data from Generation 0 model
REAL_DATA_PATH = "D_mixed.txt"                  # Original 1,518 real articles
INITIAL_MODEL = "models/MM1"                    # Generation 0 model (or "refipsai/MM1")
OUTPUT_MODEL_DIR = "models/MMP2"                # Output: Preservation Generation 1 model
N_PRESERVED = 152                               # 10% of 1,518 articles
RANDOM_SEED = 43

# ============================================================
# Helper functions
# ============================================================
def read_articles_from_text(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    raw_articles = content.split('"title: ')
    articles = []
    for raw_article in raw_articles[1:]:
        try:
            title, body = raw_article.split("body: ", 1)
            title = title.strip()
            body = body.strip()
            formatted_text = f"title: {title}\nbody: {body}"
            articles.append({"title": title, "body": body, "formatted": formatted_text})
        except ValueError:
            continue
    return pd.DataFrame(articles)


# ============================================================
# Combine synthetic data with preserved real articles, then fine-tune
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(INITIAL_MODEL)
model = AutoModelForCausalLM.from_pretrained(INITIAL_MODEL)
model.to(device)

synthetic_data = read_articles_from_text(SYNTHETIC_DATA_PATH)
original_data = read_articles_from_text(REAL_DATA_PATH)
random_subset = original_data.sample(n=N_PRESERVED, random_state=RANDOM_SEED)

combined_data = pd.concat([synthetic_data, random_subset], ignore_index=True)
combined_file_path = "combined_training_data_MMP2.txt"

with open(combined_file_path, "w") as f:
    for article in combined_data["formatted"]:
        f.write(f'"{article}"\n\n')

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
print(f"Preservation Generation 1 model saved to {OUTPUT_MODEL_DIR}")
