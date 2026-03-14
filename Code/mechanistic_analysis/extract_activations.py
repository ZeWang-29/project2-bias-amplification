# Part 1: Extract Activation Values and Save to Files
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, GPT2Tokenizer

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model names
model_names = [
    *['refipsai/MMN{}'.format(i) for i in range(2, 12)],
]

# Load the base tokenizer (GPT-2) and set padding token
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Function to extract activations
def get_activation(name, activations_dict):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations = output.detach().cpu().numpy()
        if name not in activations_dict:
            activations_dict[name] = activations
        else:
            # Ensure activations have consistent dimensions before concatenating
            if activations.shape[1] != activations_dict[name].shape[1]:
                min_tokens = min(activations.shape[1], activations_dict[name].shape[1])
                activations = activations[:, :min_tokens, :]
                activations_dict[name] = activations_dict[name][:, :min_tokens, :]
            activations_dict[name] = np.concatenate((activations_dict[name], activations), axis=0)  # Concatenate activations across articles
    return hook

# Function to read articles from text file
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
            articles.append({'formatted': formatted_text})
        except ValueError:
            continue
    return pd.DataFrame(articles)

# Load the common input file 'DD.txt' for all models
dd_text_path = '/kaggle/input/dataset-dd-new/DD.txt'
if not os.path.exists(dd_text_path):
    print(f"Input file {dd_text_path} not found.")
    exit(1)

# Read articles from the text file
articles_df = read_articles_from_text(dd_text_path)

# Initialize a list to collect all activations across models
all_activations = []

# Process each model
for model_name in model_names:
    print(f"Processing model: {model_name}")
    current_model_name = model_name

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Register hooks to capture activations
    activations_dict = {}
    hooks = []
    for i, layer in enumerate(model.transformer.h):
        layer_name = f"layer_{i}"
        hook = layer.register_forward_hook(get_activation(layer_name, activations_dict))
        hooks.append(hook)

    # Process the articles sequentially, truncated to 512 tokens
    for _, article in articles_df.iterrows():
        inputs = gpt2_tokenizer(
            article['formatted'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask, use_cache=False)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Calculate the average activation per token for each neuron at each layer
    num_articles = len(articles_df)
    num_tokens = 512
    for layer_name, activation in activations_dict.items():
        if activation.shape[1] != num_tokens:
            # Pad or truncate to ensure consistent token length
            activation = np.pad(activation, ((0, 0), (0, num_tokens - activation.shape[1]), (0, 0)), mode='constant') if activation.shape[1] < num_tokens else activation[:, :num_tokens, :]
        avg_activations = activation.sum(axis=(0, 1)) / (num_articles * num_tokens)  # Average over articles and tokens
        for neuron_id, neuron_activation in enumerate(avg_activations):
            all_activations.append({
                'model_name': model_name,
                'layer': layer_name,
                'neuron_id': neuron_id,
                'activation': neuron_activation
            })

    print(f"Collected activations for model {model_name}.")

    # Clean up memory
    del model
    torch.cuda.empty_cache()

# Create a DataFrame from the collected activations
activations_df = pd.DataFrame(all_activations)

# Save the DataFrame to a CSV file
activation_save_path = 'MMN_activations.csv'
activations_df.to_csv(activation_save_path, index=False)
print(f"All activations saved to {activation_save_path}.")
