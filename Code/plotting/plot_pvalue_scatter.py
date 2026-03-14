import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Load the neuron activation data from six CSV files and concatenate them into one DataFrame
csv_files = ['/kaggle/input/neurons-correlation/MMA_activations.csv', '/kaggle/input/neurons-correlation/MMB_activations.csv', '/kaggle/input/neurons-correlation/MMN_activations.csv', '/kaggle/input/neurons-correlation/MMO_activations.csv', '/kaggle/input/neurons-correlation/MMP_activations.csv', '/kaggle/input/neurons-correlation/MM_activations.csv']
activation_data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Load the model bias data
bias_data = pd.read_csv('/kaggle/input/newnewnewbias/ModelBias.csv')

# Merge activation data with bias data on model_name
merged_data = pd.merge(activation_data, bias_data, on='model_name')

# Prepare an empty list to store results
results = []

# Group by neuron_id and layer
grouped = merged_data.groupby(['neuron_id', 'layer'])

# Iterate through each group and calculate Pearson correlation with Newey-West adjusted standard errors
for (neuron_id, layer), group in grouped:
    # Calculate Pearson correlation between activation and bias
    correlation, _ = pearsonr(group['activation'], group['bias'])
    
    # Fit OLS model to get Newey-West adjusted p-value
    X = sm.add_constant(group['activation'])
    model = sm.OLS(group['bias'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    p_value_newey_west = model.pvalues.iloc[1]  # Get the p-value for the activation coefficient
    
    # Append the result to the list
    results.append({'neuron_id': neuron_id, 'layer': layer, 'pearson_correlation': correlation, 'significance_level_newey_west': p_value_newey_west})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
results_df.to_csv('/kaggle/working/neuron_bias_correlation_newey_west.csv', index=False)

print("Correlation calculation with Newey-West adjustment completed and results saved to 'neuron_bias_correlation_newey_west.csv'")

# Load the optimized correlation data from CSV file
correlation_df = pd.read_csv('/kaggle/working/neuron_bias_correlation_newey_west.csv')

# Plot scatter plots of p-values for each layer
plt.figure(figsize=(15, 10))

# Iterate over unique layers and create scatter plots
unique_layers = sorted(correlation_df['layer'].unique(), key=lambda x: int(x.split('_')[1]))
for layer in unique_layers:
    # Filter data for the current layer
    layer_data = correlation_df[correlation_df['layer'] == layer]
    
    # Scatter plot of neuron_id vs p-value for the current layer
    plt.scatter(layer_data['neuron_id'], layer_data['significance_level_newey_west'], label=layer, alpha=0.5)

# Plot settings
plt.axhline(y=0.05/9216, color='red', linestyle='--', label='Significance Level (0.05/9216)')
plt.xlabel("Neuron ID")
plt.ylabel("P-Value (Newey-West Adjusted)")
plt.title("Newey-West Adjusted P-Values for the Correlation Between Neuron Activations and Bias Performance")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("scatter_p_values_newey_west.png", bbox_inches='tight', dpi=300)
plt.show()
