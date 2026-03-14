# Mechanistic Interpretation Data

## Files

| File | Description |
|------|-------------|
| `Pearson Correlation Between Neuron Weight and Bias Performance.csv` | Pearson correlation values between each of the 9,216 neuron weights and the model's bias performance across 66 fine-tuned GPT-2 versions |
| `Pearson Correlation Between Neuron Weight and Text Quality Index.csv` | Pearson correlation values between each neuron weight and the text quality index |
| `Regression and Statistical Tests for Relationship Between Neuron Weight and Bias Performance.csv` | Newey-West adjusted regression results (coefficients, standard errors, p-values) for neuron weights vs. bias performance |
| `Regression and Statistical Tests for Relationship Between Neuron Weight and Generation Quality.csv` | Newey-West adjusted regression results for neuron weights vs. generation quality |

## Reproducibility

These CSV datasets contain all necessary information to reproduce:

- **Figure 4**: Pearson correlations between neuron weights and (a) bias performance, (b) generation quality
- **Section 4.4**: Mechanistic interpretation results, including identification of 3,243 bias-correlated neurons and 1,033 quality-correlated neurons with 389 overlap
