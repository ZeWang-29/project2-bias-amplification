import pandas as pd

# Load the dataset, handle bad lines
df = pd.read_csv('/kaggle/input/dataset-allsides/data_public.csv', on_bad_lines='skip')

# Define the biases and number of articles to sample from each
biases = ['From the Left', 'From the Center', 'From the Right']
n_samples = 506

# Initialize an empty list to store sampled DataFrames
sampled_dfs = []

for bias in biases:
    # Filter articles with the current bias
    df_bias = df[df['bias'] == bias].copy()
    
    # Check if there are enough articles to sample
    if len(df_bias) < n_samples:
        raise ValueError(f"Not enough articles for bias {bias} to sample {n_samples} articles.")
    
    # Randomly sample n_samples articles
    df_sampled = df_bias.sample(n=n_samples, random_state=42)
    
    # Add to the list of sampled DataFrames
    sampled_dfs.append(df_sampled)

# Combine all sampled DataFrames into one
df_combined = pd.concat(sampled_dfs, ignore_index=True)

# Shuffle the combined DataFrame to mix articles from different biases
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title and body into the desired format
def format_article(row):
    return f"title: {row['original_title']}\nbody: {row['original_body']}"

# Apply the format function to each row
df_combined['formatted'] = df_combined.apply(format_article, axis=1)

# Select only the formatted content
final_df = df_combined[['formatted']]

# Save the cleaned and formatted dataset to a new text file
final_df['formatted'].to_csv('D_mixed.txt', index=False, header=False)
