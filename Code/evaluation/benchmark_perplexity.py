# Perplexity with repetition

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import re



# Function to compute perplexity for entire articles
def compute_perplexities_whole(model, tokenizer, texts, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    perplexities = []

    for text in tqdm(texts, desc="Computing perplexity"):
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=model.config.n_positions)
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            perplexity = torch.exp(torch.tensor(loss)).item()
            perplexities.append(perplexity)

    return perplexities

# Setup your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Main analysis loop
results = pd.DataFrame()

for i in range(0, 12):  # Assuming 8 generations
    path = f'/kaggle/input/experiment1-center-9-10/D{i}.txt'  # Update path as necessary
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into articles
    articles = content.split('\n\n')
    articles = [article.strip() for article in articles if article.strip()]
    
    # Compute perplexities for each article
    perplexities = compute_perplexities_whole(model, tokenizer, articles, device)

    # Store the results in a DataFrame
    gen_results = pd.DataFrame({
        'Generation': [i] * len(perplexities),
        'Perplexity': perplexities
    })
    results = pd.concat([results, gen_results], ignore_index=True)
    
    # Saving the DataFrame to a CSV file
results.to_csv('perplexity_with_repetition.csv', index=False)
