import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

# File paths for the datasets
file_path1 = '/kaggle/input/datasetsnew-classifier/DD.classifier_scores.csv'           # Synthetic 100%
file_path2 = '/kaggle/input/datasetsnew-classifier/DDP.classifier_scores.csv'    # Synthetic 80% with Real 20%
file_path3 = '/kaggle/input/datasetsnew-classifier/DDA_classifier_scores.csv'    # Synthetic 90% with Real 10%
file_path4 = '/kaggle/input/datasetsnew-classifier/DDO_classifier_scores.csv'    # Synthetic 90% with Real 10%
file_path5 = '/kaggle/input/datasetsnew-classifier/DDN_classifier_scores.csv'    # Synthetic 90% with Real 10%
file_path6 = '/kaggle/input/datasetsnew-classifier/DDB_classifier_scores.csv'    # Synthetic 90% with Real 10%

# Load datasets
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)
df4 = pd.read_csv(file_path4)
df5 = pd.read_csv(file_path5)
df6 = pd.read_csv(file_path6)

# Assign labels to each dataset
df1['Dataset'] = 'Synthetic'
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

    # Assign a bias label to each article based on the highest score
    df['Bias_Label'] = df[['Left_Score', 'Center_Score', 'Right_Score']].idxmax(axis=1).str.replace('_Score', '')

    # Calculate the percentage of "Right" biased articles for each generation
    right_percentage = df[df['Bias_Label'] == 'Right'].groupby('Generation').size() / df.groupby('Generation').size() * 100

    # Sort the percentages by generation
    right_percentage = right_percentage.sort_index()

    # Plotting the percentage of Right biased articles across generations for each dataset
    plt.plot(right_percentage.index, right_percentage, marker='o', linestyle='-', label=label)

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

# Assuming the generations are consistent across datasets, use any of the right_percentage indexes for xticks
plt.xticks(right_percentage.index, [generation_labels.get(gen, f"Generation {gen}") for gen in right_percentage.index], rotation=45, ha='right')

# Adding title and labels
plt.title('Percentage of Right Biased Articles Across Generations for Different Datasets')
plt.xlabel('Generation')
plt.ylabel('Percentage of Right Biased Articles (%)')
plt.legend(title='Dataset')
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig('percentage_right_biased.png', dpi=300)

# Display the plot
plt.show()
