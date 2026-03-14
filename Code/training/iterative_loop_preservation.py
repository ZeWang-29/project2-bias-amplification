"""
Iterative synthetic fine-tuning loop with the Preservation mitigation strategy.

Same as the baseline loop, but at each generation, 10% (152) randomly selected
real articles are added to the synthetic training data.

Paper reference: Section 4.3 (Mitigation Strategies — Preservation)
"""

import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Configuration — adjust these paths to your environment
# ============================================================
REAL_DATA_PATH = "D_mixed.txt"                  # Original 1,518 real articles
INITIAL_MODEL = "models/MMP2"                   # Preservation Generation 1 model (or "refipsai/MMP2")
OUTPUT_DIR = "models"
SYNTHETIC_DIR = "synthetic_data"
START_GEN = 2                                   # First generation to produce (inclusive)
END_GEN = 10                                    # Last generation to produce (inclusive)
BLOCK_SIZE = 64
N_PRESERVED = 152                               # 10% of 1,518 articles

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


def split_title_and_body(text):
    title, body = text.split("body: ", 1)
    title = title.replace("title: ", "").strip()
    body = body.strip()
    return title, body


def generate_synthetic_article(examples, max_leng, tokenizer, model):
    synthetic_articles = []
    for text in examples["formatted"]:
        title, body = split_title_and_body(text)
        title_tokens = tokenizer.encode(title, add_special_tokens=False)
        body_tokens = tokenizer.encode(body, add_special_tokens=False)
        blocks = [title_tokens]
        for i in range(0, len(body_tokens), max_leng):
            blocks.append(body_tokens[i:i + max_leng])
        synthetic_tokens = []
        for block in blocks:
            input_ids = torch.tensor([block]).to(model.device)
            attention_mask = torch.ones(input_ids.shape, device=model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=len(block) + max_leng,
                    do_sample=False,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_sequence = outputs[0][len(block):].cpu().numpy()
            synthetic_tokens.extend(generated_sequence)
        synthetic_article = tokenizer.decode(synthetic_tokens, skip_special_tokens=True)
        synthetic_articles.append(synthetic_article)
    return {"synthetic_article": synthetic_articles}


def fine_tune_model_with_preservation(synthetic_data_path, real_data_path, output_dir,
                                      tokenizer, model, iteration):
    """Fine-tune on synthetic data + 10% randomly preserved real articles."""
    synthetic_data = read_articles_from_text(synthetic_data_path)
    original_data = read_articles_from_text(real_data_path)
    random_subset = original_data.sample(n=N_PRESERVED, random_state=42 + iteration)
    combined_data = pd.concat([synthetic_data, random_subset], ignore_index=True)

    combined_file_path = f"combined_training_data_MMP{iteration + 1}.txt"
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
        output_dir=output_dir,
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
        model=model, args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator, tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ============================================================
# Main iterative loop
# ============================================================
current_model_name = INITIAL_MODEL

for gen in tqdm(range(START_GEN, END_GEN + 1), desc="Generations"):
    print(f"\n=== Generation {gen} (Preservation) ===")

    tokenizer = AutoTokenizer.from_pretrained(current_model_name)
    model = AutoModelForCausalLM.from_pretrained(current_model_name)
    model.to(device)

    synthetic_output_path = f"{SYNTHETIC_DIR}/DDP{gen}.txt"
    model_output_dir = f"{OUTPUT_DIR}/MMP{gen + 1}"

    # Step 1: Generate synthetic articles
    df = read_articles_from_text(REAL_DATA_PATH)
    dataset = Dataset.from_pandas(df)

    def map_function(examples):
        return generate_synthetic_article(examples, BLOCK_SIZE, tokenizer, model)

    processed_dataset = dataset.map(map_function, batched=True, batch_size=124)
    df_synthetic = processed_dataset.to_pandas()

    df_synthetic["formatted_synthetic"] = df_synthetic.apply(
        lambda row: f'"title: {row["title"]}\nbody: {row["synthetic_article"]}"', axis=1
    )
    with open(synthetic_output_path, "w") as f:
        for article in df_synthetic["formatted_synthetic"]:
            f.write(article + "\n\n")

    # Step 2: Fine-tune with preservation (synthetic + 10% real)
    fine_tune_model_with_preservation(
        synthetic_output_path, REAL_DATA_PATH, model_output_dir, tokenizer, model, gen
    )
    print(f"  Model saved to {model_output_dir}")

    current_model_name = model_output_dir
