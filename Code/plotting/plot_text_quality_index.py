import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

# File paths for the datasets
file_path1 = '/kaggle/input/dataset-gibberish/DD.gibberish_levels.csv'           # Synthetic 100%
file_path2 = '/kaggle/input/dataset-gibberish/DDP.gibberish_levels.csv'    # Synthetic 80% with Real 20%
file_path3 = '/kaggle/input/dataset-gibberish/DDA_gibberish_levels.csv'    # Synthetic 90% with Real 10%
file_path4 = '/kaggle/input/dataset-gibberish/DDO_gibberish_levels.csv'    # Synthetic 90% with Real 10%
file_path5 = '/kaggle/input/dataset-gibberish/DDN_gibberish_levels.csv'    # Synthetic 90% with Real 10%
file_path6 = '/kaggle/input/dataset-gibberish/DDB_gibberish_levels.csv'    # Synthetic 90% with Real 10%

# Load datasets
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)
df4 = pd.read_csv(file_path4)
df5 = pd.read_csv(file_path5)
df6 = pd.read_csv(file_path6)

# Assign labels to each dataset
# Skip rows where Generation is 0
# Skip rows where Generation is 0
df1 = df1[df1['Generation'] != 0]

# Add the 'Dataset' column with the value 'Synthetic'
df1['Dataset'] = 'Synthetic'


# Skip rows where Generation is 0
df2 = df2[df2['Generation'] != 0]


df2['Dataset'] = 'Synthetic with Preservation'
df3['Dataset'] = 'Synthetic with Accumulation'
df4['Dataset'] = 'Synthetic with Overfitting'
df5['Dataset'] = 'Synthetic with Nucleus Sampling'
df6['Dataset'] = 'Synthetic with Beam Search'

# Combine all datasets into a list for easy processing
datasets = [(df1, 'Synthetic'),
            (df2, 'Synthetic with Preservation'),
            (df3, 'Synthetic with Accumulation'),
            (df4, 'Synthetic with Overfitting'),
            (df5, 'Synthetic with Nucleus Sampling'),
            (df6, 'Synthetic with Beam Search')]

plt.figure(figsize=(12, 8))

# Process each dataset separately
for df, label in datasets:
    # Convert the 'Generation' column to integers for correct sorting
    df['Generation'] = df['Generation'].astype(int)

    # Group by 'Generation' and calculate the mean and standard deviation of perplexity
    avg_perplexity = df.groupby('Generation')['GibberishLevel'].mean()
    std_perplexity = df.groupby('Generation')['GibberishLevel'].std()
    count = df.groupby('Generation')['GibberishLevel'].size()
    
    # Calculate the confidence intervals
    ci = 1.96 * (std_perplexity / np.sqrt(count))

    # Plotting the average perplexity across generations for each dataset with confidence intervals
    plt.errorbar(
        avg_perplexity.index,
        avg_perplexity.values,
        yerr=ci.values,
        marker='o',  # Keeping the marker consistent with the classifier plot
        capsize=5,   # Size of the caps on error bars
        label=label,
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

plt.xticks(avg_perplexity.index, [generation_labels.get(gen, f"Generation {gen}") for gen in avg_perplexity.index], rotation=45, ha='right')

# Adding title and labels
plt.title('Average Text Quality Index Across Generations')
plt.xlabel('Generation')
plt.ylabel('Text Quality Index')
plt.legend(title='Dataset')
plt.grid(True)
plt.tight_layout()

plt.savefig('average_perplexity_with_confidence_intervals.png', dpi=300)

# Display the plot
plt.show()
