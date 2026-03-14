### import pandas as pd
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
    articles = []
    current_article = []
    
    try:
        # Open the file in read mode with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # If the line starts with '"title:', it indicates the start of a new article
                if line.strip().startswith('"title:'):
                    # If there's an existing article, append it to the articles list
                    if current_article:
                        article_body = ''.join(current_article).strip()  # Combine the lines into one article body
                        articles.append({'body': article_body, 'formatted': article_body})
                        current_article = []  # Reset for the next article

                # Add the line to the current article (even if it's the title line, we add everything)
                current_article.append(line)
            
            # Append the last article in case the file doesn't end with a new "title:" line
            if current_article:
                article_body = ''.join(current_article).strip()
                articles.append({'body': article_body, 'formatted': article_body})
                
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except UnicodeDecodeError as ude:
        print(f"Encoding error: {ude}. Try using a different encoding.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return pd.DataFrame(articles)


# Function to split title and body from formatted text
def split_title_and_body(text):
    title, body = text.split('body: ', 1)
    title = title.replace('title: ', '').strip()
    body = body.strip()
    return title, body



# Function to fine-tune the model
def fine_tune_model(train_file_paths, output_dir):
    
    dataframes = []

# Iterate over the list of file paths and read each dataset
    for file_path in train_file_paths:
    # Read the data from each file
        data = read_articles_from_text(file_path)
    # Add the DataFrame to the list
        dataframes.append(data)
        
    combined_data = pd.concat(dataframes, ignore_index=True)
    combined_file_path = '/kaggle/working/CombinedTrainingDataforMMA2.txt'

    with open(combined_file_path, 'w') as f:
        for article in combined_data['formatted']:
            f.write(f"{article}\n\n")
  
    dataset = load_dataset('text', data_files={'train': combined_file_path})


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
previous_datasets = [f'/kaggle/input/dda-and-dda1/DDA.txt', f'/kaggle/input/dda-and-dda1/DDA1.txt' ]  # Start with D0 and D1
current_model_name = 'refipsai/MM1'  
rows = []


max_leng = 64
tokenizer = AutoTokenizer.from_pretrained(current_model_name)
model = AutoModelForCausalLM.from_pretrained(current_model_name)
model.to(device)

model_output_dir = f'/kaggle/working/MMA{2}'

fine_tune_model(previous_datasets, model_output_dir)

current_model_name = model_output_dir
