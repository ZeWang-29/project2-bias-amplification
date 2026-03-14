import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV dataset
df = pd.read_csv('/kaggle/input/datasets-classifier/DD.classifier_scores.csv')

# Determine the bias label for each article
def classify_bias(row):
    scores = {'Center': row['Center_Score'],
              'Right': row['Right_Score'],
              'Left': row['Left_Score']}
    return max(scores, key=scores.get)

df['Bias_Label'] = df.apply(classify_bias, axis=1)

# Convert 'Generation' to string to handle 'initial' and numeric generations uniformly
df['Generation'] = df['Generation'].astype(str)

# Filter the dataframe to only include Generation 0 ('initial')
df_generation_0 = df[df['Generation'] == '0']

# Count the number of articles for each label in Generation 0
bias_counts = df_generation_0['Bias_Label'].value_counts().reindex(['Left', 'Center', 'Right'], fill_value=0)

# Calculate the percentage of articles for each label
bias_percentages = (bias_counts / bias_counts.sum()) * 100

# Plot for Generation 0
plt.figure(figsize=(10, 8))
bias_counts.plot(kind='bar', color=['blue', 'grey', 'red'], width=0.6)
plt.title('Distribution of Articles in GPT-2 Outputs by Political Leaning ', fontsize=15)
plt.xlabel('', fontsize=15)
plt.ylabel('Number of Articles', fontsize=15)
plt.xticks(rotation=0, fontsize=15)
plt.ylim(0, bias_counts.max() + 50)
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

plt.tight_layout()
plt.savefig('bias_distribution.png')
plt.yticks(fontsize=15)
plt.show()