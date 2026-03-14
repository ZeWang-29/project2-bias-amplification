import numpy as np 
from scipy.optimize import minimize
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn theme for better aesthetics
sns.set(style="whitegrid")

# Define explicit color mapping for generations
generation_colors = {
    1: 'green',
    5: 'blue',
    7: 'purple',
    11: 'orange'
}

# Step 1: Pretraining Phase - Generate data from Beta(3, 2) and estimate parameters
alpha_pretrain = 3
beta_pretrain = 2
data_pretrain = np.random.beta(alpha_pretrain, beta_pretrain, size=10000)

def neg_log_likelihood_beta(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    return -np.sum(np.log(beta.pdf(data, a, b)))

initial_params = [1.0, 1.0]
bounds = [(1e-6, None), (1e-6, None)]

result_pretrain = minimize(neg_log_likelihood_beta, initial_params, args=(data_pretrain,), bounds=bounds)
alpha_est_pretrain, beta_est_pretrain = result_pretrain.x
print(f"Pretraining MLE estimates: α = {alpha_est_pretrain:.2f}, β = {beta_est_pretrain:.2f}")

# Step 2: Function to compute weights based on the estimated parameters
def compute_weights(data, alpha_est, beta_est):
    return beta.pdf(data, alpha_est, beta_est)

# Step 3: Fine-tuning Phase - Generate data from the true distribution Beta(2, 2)
alpha_true = 2
beta_true = 2
data_finetune = np.random.beta(alpha_true, beta_true, size=10000)

# Step 4: Perform weighted MLE on the fine-tuning data (Generations 1 to 11)
def neg_log_likelihood_beta_weighted(params, data, weights):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    log_pdf = np.log(beta.pdf(data, a, b))
    return -np.sum(weights * log_pdf)

generations = 11
data_synthetic = data_finetune.copy()  # Start with the true Beta(2,2) data
generations_data = []

initial_params = [2.0, 2.0]  # Initial guess for MLE

# Step 5: Iterate to generate synthetic data for Generations 1 to 11
for gen in range(1, generations + 1):
    # Compute weights dynamically based on the synthetic data from the previous generation
    weights = compute_weights(data_synthetic, alpha_est_pretrain, beta_est_pretrain)
    
    # Perform weighted MLE using the updated weights from synthetic data
    result = minimize(neg_log_likelihood_beta_weighted, initial_params, args=(data_synthetic, weights), bounds=bounds)
    alpha_est, beta_est = result.x
    print(f"Generation {gen} MLE estimates: α = {alpha_est:.2f}, β = {beta_est:.2f}")

    # Generate synthetic data based on the new estimates (Dn)
    data_synthetic = np.random.beta(alpha_est, beta_est, size=10000)

    # Store data for specific generations (1, 5, 7, 11)
    if gen in [1, 5, 7, 11]:
        generations_data.append((gen, data_synthetic.copy(), alpha_est, beta_est))

    # Update the pretraining estimates with the current generation's estimates
    alpha_est_pretrain, beta_est_pretrain = alpha_est, beta_est

    # Update initial parameters for the next iteration
    initial_params = [alpha_est, beta_est]

# Step 6: Plot the true Beta(2,2) distribution, pretraining estimated Beta, and the estimated generations
x = np.linspace(0, 1, 100)

plt.figure(figsize=(12, 8))

# Plot the pretraining estimated Beta(3,2) distribution in black
plt.plot(
    x, 
    beta.pdf(x, alpha_pretrain, beta_pretrain), 
    color='black', 
    linestyle='-', 
    lw=2, 
    label=f'Beta(3,2)'
)

# Plot the true balanced Beta(2,2) distribution in red
plt.plot(
    x, 
    beta.pdf(x, alpha_true, beta_true), 
    color='red', 
    linestyle='--', 
    lw=2, 
    label='Beta(2,2)'
)

# Define labels for specific generations
generation_labels = {
    1: "Generation 0",
    5: "Generation 4",
    7: "Generation 6",
    11: "Generation 10"
}

# Plot each selected generation with its assigned color
for gen, data_gen, alpha_est, beta_est in generations_data:
    label = generation_labels.get(gen, f"Generation {gen}")
    color = generation_colors.get(gen, 'gray')  # Default to gray if not defined
    sns.kdeplot(
        data_gen, 
        bw_adjust=0.5, 
        label=label, 
        color=color, 
        linewidth=2
    )

plt.title('Beta Distributions Estimation, with WMLE', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.savefig('/kaggle/working/WMLE_improved.png')
plt.show()
