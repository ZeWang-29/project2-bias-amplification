import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data from the CSV files
file_path1 = '/kaggle/working/perplexity_without_repetition.csv'           # Synthetic 100%


df1 = pd.read_csv(file_path1)

# Add a 'Dataset' column to each DataFrame with the labels
df1['Dataset'] = 'Synthetic 100%'


# Ensure that 'Generation' column is integer type
df1['Generation'] = df1['Generation'].astype(int)

# Combine the three datasets for the swarm plot

# Function to calculate average scores and confidence intervals for a dataset
def compute_generation_stats(df):
    grouped = df.groupby('Generation')
    avg_scores = grouped['Perplexity'].mean()
    std_scores = grouped['Perplexity'].std()
    counts = grouped.size()
    standard_errors = std_scores / np.sqrt(counts)
    confidence_intervals = 1.96 * standard_errors  # For 95% confidence interval
    return avg_scores, confidence_intervals

# Compute stats for all datasets
avg_scores1, ci1 = compute_generation_stats(df1)


# Plotting average GibberishLevel across generations for all three datasets
plt.figure(figsize=(14, 8))

# Plot dataset 1
plt.errorbar(
    avg_scores1.index,
    avg_scores1.values,
    yerr=ci1.values,
    marker='o',
    capsize=5,
    label='Synthetic 100%',
    color='blue',
    linestyle='-'
)


# Replace the generation numbers with custom labels on the x-axis
generation_labels = {
    0: "GPT-2",
    1: "Generation 0",
    2: "Generation 1",  
    3: "Generation 2",
    4: "Generation 3",
    5: "Generation 4",
    6: "Generation 5",
    7: "Generation 6",
    8: "Generation 7",
    9: "Generation 8",
    10: "Generation 9",
    11: "Generation 10"
}

plt.xticks(avg_scores1.index, [generation_labels.get(gen, f"Generation {gen}") for gen in avg_scores1.index], rotation=45, ha='right')

# Customizing labels and title
plt.title('Average Text Quality Across Generations')
plt.xlabel('Generation')
plt.ylabel('Average Text Quality')
plt.legend()
plt.grid(False)
plt.tight_layout()  # Adjust layout to ensure labels fit well

# Save the plot to a file
plt.savefig('average_perplexity.png', dpi=300)

# Display the plot
plt.show()
