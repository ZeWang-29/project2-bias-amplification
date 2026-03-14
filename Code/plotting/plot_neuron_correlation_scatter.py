import pandas as pd
import matplotlib.pyplot as plt

# Load the optimized correlation data from CSV file
correlation_df = pd.read_csv("/kaggle/working/neuron_performance_correlation.csv")

# Report neurons with correlation >= 0.80 or <= -0.80
significant_neurons = correlation_df[(correlation_df["pearson_correlation"] >= 0.80) | (correlation_df["pearson_correlation"] <= -0.80)]
significant_neurons.to_csv("significant_neurons.csv", index=False)
print("Significant neurons saved to significant_neurons.csv")

# Plot scatter plots of correlations for each layer
plt.figure(figsize=(15, 10))

# Iterate over unique layers and create scatter plots
unique_layers = sorted(correlation_df["layer"].unique(), key=lambda x: int(x.split('_')[1]))
for layer in unique_layers:
    # Filter data for the current layer
    layer_data = correlation_df[correlation_df["layer"] == layer]
    
    # Scatter plot of neuron_id vs spearman_correlation for the current layer
    plt.scatter(layer_data["neuron_id"], layer_data["pearson_correlation"], label=layer, alpha=0.5)

# Plot settings
plt.xlabel("Neuron ID")
plt.ylabel("Correlation Level")
plt.title("Pearson Correlation Between Neuron Activations and Political Bias")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("scatter_pearson_correlations.png", bbox_inches='tight', dpi=300)
plt.show()
