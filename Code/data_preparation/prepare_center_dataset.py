import pandas as pd

# Load the dataset, read data from allsides articles dataset: https://webis.de/data/webis-bias-flipper-18.html
df = pd.read_csv('data_public.csv', on_bad_lines='skip')

# Filter only "center" articles and create a copy
df_center = df[df['bias'] == 'From the Center'].copy()

# Combine title and body into the desired format
def format_article(row):
    return f"title: {row['original_title']}\nbody: {row['original_body']}"

# Apply the format function to each row
df_center['formatted'] = df_center.apply(format_article, axis=1)

# Select only the formatted content
final_df = df_center[['formatted']]

# Save the cleaned and formatted dataset to a new text file
final_df['formatted'].to_csv('D0.txt', index=False, header=False)
