import pandas as pd
from transformers import pipeline, Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer, \
    DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to read and structure articles from the text file
def read_articles_from_text(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    raw_articles = content.split('"title: ')
    articles = []
    for raw_article in raw_articles[1:]:
        try:
            title, body = raw_article.split('body: ', 1)
            title = title.strip()
            body = body.strip()
            formatted_text = f"title: {title}\nbody: {body}"
            articles.append({'title': title, 'body': body, 'formatted': formatted_text})
        except ValueError:
            continue
    return pd.DataFrame(articles)


# Function to split title and body from formatted text
def split_title_and_body(text):
    title, body = text.split('body: ', 1)
    title = title.replace('title: ', '').strip()
    body = body.strip()
    return title, body



# Function to fine-tune the model
def fine_tune_model(train_file_path, output_dir):
    # Load the prepared dataset
    dataset = load_dataset('text', data_files={'train': train_file_path})

    # Initialize the tokenizer and model

    # Set the EOS token as the padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Set up the data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,  # Number of epochs
        per_device_train_batch_size=8,  # Batch size per device during training
        save_steps=10_000,  # Save the model every 10,000 steps
        save_total_limit=2,  # Limit the total amount of checkpoints
        logging_dir='./logs',  # Directory for storing logs
        logging_steps=200,  # Log every 200 steps
        learning_rate=5e-5,  # Learning rate
        weight_decay=0.01,  # Weight decay
        evaluation_strategy="no",
        report_to="none"  # Disable W&B reporting
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start fine-tuning
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)




# Instead of using append, collect all rows and concatenate them at the end
rows = []


max_leng = 64
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)

# Define file paths
synthetic_output_file_path = f'/kaggle/input/d-mixed/D_mixed.txt'
model_output_dir = f'/kaggle/working/MM1'




# Fine-tune the model with the new synthetic data (D2, D3, ..., D10)
fine_tune_model(synthetic_output_file_path, model_output_dir)

# Update the current model to the newly fine-tuned model for the next iteration
current_model_name = model_output_dir
