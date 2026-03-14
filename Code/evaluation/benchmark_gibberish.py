#Gibberish Code

import torch
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer

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

    return articles

# Function to compute gibberish levels for entire articles
def compute_gibberish_levels(texts, model_name="madhurjindal/autonlp-Gibberish-Detector-492513457", max_length=512):
    classifier = pipeline("text-classification", model=model_name, top_k=4, device=0)  # Use GPU if available
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_map = {
        'noise': 0,
        'word salad': 1,
        'mild gibberish': 2,
        'clean': 3
    }
    gibberish_levels = []
    total_articles = len(texts)
    for index, text in enumerate(texts):
        sentences = text['body'].split('.')
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        article_levels = []
        for sentence in sentences:
            # Tokenize the sentence and truncate if it's longer than max_length
            inputs = tokenizer(sentence, truncation=True, max_length=max_length, return_tensors="pt")
            # Move tensors to the appropriate device
            inputs = {key: value.to(classifier.device) for key, value in inputs.items()}
            # Disable gradient computation
            with torch.no_grad():
                outputs = classifier.model(**inputs)
                logits = outputs.logits
                scores = logits.softmax(dim=-1)
                results = [{'label': classifier.model.config.id2label[label.item()], 'score': score.item()} for label, score in zip(logits.argmax(dim=-1), scores.max(dim=-1))]
            # Compute expected gibberish level
            expected_level = sum([label_map[res['label']] * res['score'] for res in results])
            article_levels.append(expected_level)
        # Calculate the average gibberish level for the article
        if article_levels:
            average_level = sum(article_levels) / len(article_levels)
        else:
            average_level = 0  # Default to 0 if no sentences
        gibberish_levels.append(average_level)
    return gibberish_levels

# Initialize the results DataFrame
results = pd.DataFrame(columns=['Generation', 'GibberishLevel'])

for i in range(0, 12):  # Assuming 8 generations
    path = f'/kaggle/input/dataset-2-center/DP{i}.txt'
    articles = read_articles_from_text(path)
    print(f"Generation {i}: {len(articles)} articles")
    unique_articles = {article['body'] for article in articles}
    print(f"Original count: {len(articles)}, Unique count: {len(unique_articles)}")
    gibberish_levels = compute_gibberish_levels(articles)
    gen_results = pd.DataFrame({
        'Generation': [i] * len(gibberish_levels),
        'GibberishLevel': gibberish_levels
    })
    results = pd.concat([results, gen_results], ignore_index=True)

# Save results to CSV
results.to_csv('gibberish_levels.csv', index=False)

# Group results by Generation and compute statistics
grouped_results = results.groupby('Generation')['GibberishLevel']
mean_levels = grouped_results.mean()
std_levels = grouped_results.std()
