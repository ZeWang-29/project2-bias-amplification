import pandas as pd
from transformers import pipeline, RobertaTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from io import StringIO
import re
import torch

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Initialize the classifier and tokenizer
classifier_model_name = "wu981526092/bias_classifier_roberta"
classifier = pipeline("text-classification", model=classifier_model_name, tokenizer=classifier_model_name, top_k=3, device=0)
classifier_tokenizer = RobertaTokenizer.from_pretrained(classifier_model_name)

# Function to truncate text to a maximum of 512 tokens
def truncate_text(text, tokenizer, max_length=500):
    encodings = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    return tokenizer.decode(encodings.input_ids[0], skip_special_tokens=True)

# Function to get classifier scores for articles
def get_classifier_scores(classifier, tokenizer, texts):
    left_scores = []
    center_scores = []
    right_scores = []
    for text in texts:
        truncated_text = truncate_text(text, tokenizer)
        try:
            # Redirect stdout temporarily
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            results = classifier(truncated_text)
            # Restore stdout
            sys.stdout = old_stdout
            # Check and flatten results if necessary
            if isinstance(results, list) and isinstance(results[0], list):
                results = results[0]
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
                scores = {result['label']: result['score'] for result in results}
                right_scores.append(scores.get('LABEL_0', 0))  # Correct label for Right
                left_scores.append(scores.get('LABEL_1', 0))   # Correct label for Left
                center_scores.append(scores.get('LABEL_2', 0)) # Correct label for Center
            else:
                print(f"Unexpected format in classifier results: {results}")
        except Exception as e:
            print(f"Error processing text: {e}")
    return left_scores, center_scores, right_scores

# Initialize lists to hold scores and article IDs across all generations
all_scores = []

# Create a list of generations, starting with 'initial'
generations = ['initial'] + list(range(0, 1))

# Loop through generations
for gen in generations:
    if gen == 'initial':
        print(f"Processing generation {gen}")
        synthetic_input_file_path = '/kaggle/working/D_cleaned.txt'
        generation_label = 'initial'
    else:
        print(f"Processing generation {gen}")
        synthetic_input_file_path = f'/kaggle/input/dataset-check/D{gen}.txt'
        generation_label = gen

    # Read the synthetic articles
    try:
        with open(synthetic_input_file_path, 'r') as file:
            synthetic_articles = file.read().split('\n\n')
        synthetic_articles = [article.replace('"', '').strip() for article in synthetic_articles if article.strip()]
        # Get classifier scores for the synthetic articles
        left_scores, center_scores, right_scores = get_classifier_scores(classifier, classifier_tokenizer, synthetic_articles)
        # Store scores with generation information
        all_scores.extend([
            {'Generation': generation_label, 'Article_ID': idx+1, 'Left_Score': left, 'Center_Score': center, 'Right_Score': right}
            for idx, (left, center, right) in enumerate(zip(left_scores, center_scores, right_scores))
        ])
    except FileNotFoundError:
        print(f"File not found: {synthetic_input_file_path}")
    except Exception as e:
        print(f"Error processing generation {gen}: {e}")

# Convert scores to DataFrame and save to CSV
scores_df = pd.DataFrame(all_scores)
scores_df.to_csv('/kaggle/working/classifier_scores.csv', index=False)
