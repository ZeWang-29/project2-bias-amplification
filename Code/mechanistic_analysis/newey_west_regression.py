import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

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
