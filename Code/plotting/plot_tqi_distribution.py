import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})

# Load the DataFrame from a CSV file
file_path = '/kaggle/input/dd-gibberish/DD.gibberish_levels.csv'  # Replace with the path to your CSV file
results = pd.read_csv(file_path)

# Grouping data by generation
grouped_perplexities = results.groupby('Generation')['GibberishLevel'].apply(list)

# Specify the generations you want to include in the plot
selected_generations = [1, 3, 5, 7, 9, 11]

# Plotting the histogram
plt.figure(figsize=(12, 6))  # Sets up the figure size

# Using a colormap
color_map = plt.get_cmap('viridis')  # Using 'viridis' for a visually appealing set of colors
colors = color_map(np.linspace(0, 1, len(selected_generations)))  # Generates colors for each generation

# Plotting each generation's perplexity histogram on the same figure
for idx, generation in enumerate(selected_generations):
    if generation in grouped_perplexities:
        # Ensuring there's data to plot for each selected generation
        perplexities = grouped_perplexities[generation]
        plt.hist(perplexities, bins=100, alpha=0.75, color=colors[idx], label=f'Generation {generation - 1}')  # Label as Generation i-1

plt.xlabel('Text Quality Index')  # Sets the label for the x-axis
plt.ylabel('Frequency')  # Sets the label for the y-axis
plt.title('Distribution of Text Quality Index Across Generations')  # Sets the title of the histogram
plt.legend()  # Adds a legend to distinguish the generations

plt.savefig('distribution_text_quality.png', dpi=300, bbox_inches='tight')  # Saves the figure

plt.show()
