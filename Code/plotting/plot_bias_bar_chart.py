import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Read the CSV dataset
df = pd.read_csv('/kaggle/working/classifier_scores.csv')

# Determine the bias label for each article
def classify_bias(row):
    scores = {'Center': row['Center_Score'],
              'Right': row['Right_Score'],
              'Left': row['Left_Score']}
    return max(scores, key=scores.get)

df['Bias_Label'] = df.apply(classify_bias, axis=1)

# Convert 'Generation' to string to handle 'initial' and numeric generations uniformly
df['Generation'] = df['Generation'].astype(str)

# Count the number of articles for each label in each generation
bias_counts = df.groupby(['Generation', 'Bias_Label']).size().unstack(fill_value=0)

# Sort generations: 'initial' first, then numeric generations in order
def sort_generations(gen_list):
    gen_list_numeric = [int(gen) for gen in gen_list if gen.isdigit()]
    gen_list_sorted = sorted(gen_list_numeric)
    if 'initial' in gen_list:
        return ['initial'] + [str(gen) for gen in gen_list_sorted]
    else:
        return [str(gen) for gen in gen_list_sorted]

generations = sort_generations(df['Generation'].unique())

bias_labels = ['Left', 'Center', 'Right']

# Calculate the number of rows and columns for subplots based on the number of generations
num_generations = len(generations)
cols = 3  # You can adjust the number of columns as needed
rows = math.ceil(num_generations / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for idx, generation in enumerate(generations):
    ax = axes[idx]
    # Ensure that the generation exists in bias_counts
    if generation in bias_counts.index:
        counts = bias_counts.loc[generation][bias_labels]
    else:
        counts = pd.Series([0, 0, 0], index=bias_labels)
    counts.plot(kind='bar', ax=ax, color=['blue', 'grey', 'red'])
    ax.set_title(f'Generation {generation}')
    ax.set_xlabel('Bias Label')
    ax.set_ylabel('Number of Articles')
    ax.set_ylim(0, bias_counts.max().max() + 10)
    ax.set_xticklabels(bias_labels, rotation=0)

# Hide any unused subplots
for idx in range(num_generations, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()
