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


def generate_synthetic_article(examples, max_leng):
    synthetic_articles = []
    
    for text in examples['formatted']:
        # Split the text into title and body
        title, body = split_title_and_body(text)

        # Tokenize the title and body separately
        title_tokens = tokenizer.encode(title, add_special_tokens=False)
        body_tokens = tokenizer.encode(body, add_special_tokens=False)

        # Prepare blocks: Title block and body blocks of max_leng tokens each
        blocks = [title_tokens]  # Title is the first block
        for i in range(0, len(body_tokens), max_leng):
            blocks.append(body_tokens[i:i + max_leng])

        # Generate synthetic tokens for each block
        synthetic_tokens = []
        for block in blocks:
            input_ids = torch.tensor([block]).to(model.device)  # Move input to GPU
            attention_mask = torch.ones(input_ids.shape, device=model.device)

            with torch.no_grad():  # Disable gradient calculation (saves memory and computation)
                outputs = model.generate(
                    input_ids,
                    max_length=len(block) + max_leng,  # Generate len(block) + max_leng tokens in total
                    do_sample=False,  # Deterministic output
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Extract only the generated portion (skip input tokens)
            generated_sequence = outputs[0][len(block):].cpu().numpy()  # Move to CPU and convert to NumPy array
            synthetic_tokens.extend(generated_sequence)

        # Convert synthetic tokens back to text and add to the list
        synthetic_article = tokenizer.decode(synthetic_tokens, skip_special_tokens=True)
        synthetic_articles.append(synthetic_article)

    return {'synthetic_article': synthetic_articles}

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



# Load the very first model trained on the real data
current_model_name = 'refipsai/MM2'

# Instead of using append, collect all rows and concatenate them at the end
rows = []

for i in tqdm(range(2, 4), desc="Processing iterations"):  # Start from M3 to M10
    print(f"Processing iteration {i}")

    max_leng = 64
    tokenizer = AutoTokenizer.from_pretrained(current_model_name)
    model = AutoModelForCausalLM.from_pretrained(current_model_name)
    model.to(device)

    # Define file paths
    input_file_path = f'/kaggle/input/d-mixed/D_mixed.txt'
    synthetic_output_file_path = f'/kaggle/working/DD{i}.txt'
    model_output_dir = f'/kaggle/working/MM{i + 1}'

    # Read and process the original data file (D0)
    df = read_articles_from_text(input_file_path)
    dataset = Dataset.from_pandas(df)


    def map_function(examples):
        return generate_synthetic_article(examples, max_leng)


    # Generate synthetic data for the next iteration (D2, D3, ..., D10)
    processed_dataset = dataset.map(map_function, batched=True, batch_size=124)
    df_synthetic = processed_dataset.to_pandas()

    # Format and save the synthetic data to the next file (D2, D3, ..., D10)
    df_synthetic['formatted_synthetic'] = df_synthetic.apply(
        lambda row: f'"title: {row["title"]}\nbody: {row["synthetic_article"]}"', axis=1
    )
    with open(synthetic_output_file_path, 'w') as f:
        for article in df_synthetic['formatted_synthetic']:
            f.write(article + '\n\n')

    # Fine-tune the model with the new synthetic data (D2, D3, ..., D10)
    fine_tune_model(synthetic_output_file_path, model_output_dir)

    # Update the current model to the newly fine-tuned model for the next iteration
    current_model_name = model_output_dir
