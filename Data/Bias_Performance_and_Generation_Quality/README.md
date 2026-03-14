# Bias Performance and Generation Quality Across Generations and Setups

## Notation
In the CSV files:
- **Generation 1** corresponds to **Generation 0** in the paper.
- **Generation 11** corresponds to **Generation 10** in the paper.

## Files

| File | Description |
|------|-------------|
| `Synthetic_Bias_Performance.csv` | Bias classifier scores for baseline (no mitigation), unbiased dataset |
| `Synthetic_Generation_Quality.csv` | Text quality index for baseline (no mitigation), unbiased dataset |
| `Synthetic_Perplexity.csv` | Perplexity scores for baseline (no mitigation), unbiased dataset |
| `Overfitting_Bias_Performance.csv` | Bias classifier scores with Overfitting mitigation |
| `Overfitting_Generation_Quality.csv` | Text quality index with Overfitting mitigation |
| `Preservation_Bias_Performance.csv` | Bias classifier scores with Preservation mitigation |
| `Preservation_Generation_Quality.csv` | Text quality index with Preservation mitigation |
| `Accumulation_Bias_Performance.csv` | Bias classifier scores with Accumulation mitigation |
| `Accumulation_Generation_Quality.csv` | Text quality index with Accumulation mitigation |
| `Synthetic_Center-Leaning-Only_Bias_Performance.csv` | Bias scores for alternative setup (center-leaning only) |
| `Synthetic_Center-Leaning-Only_Generation_Quality.csv` | Text quality for alternative setup (center-leaning only) |
| `Preservation_Center-Leaning-Only_Bias_Performance.csv` | Bias scores for alternative setup with Preservation |
| `Preservation_Center-Leaning-Only_Generation_Quality.csv` | Text quality for alternative setup with Preservation |
| `GPT2_Bias_Performance.csv` | Bias classifier scores for base GPT-2 (pre-fine-tuning) |

## Reproducibility

These CSV datasets contain all necessary information to reproduce the following figures from the paper:

- **Figure 2**: Distribution of political bias labels for initial GPT-2 outputs (`GPT2_Bias_Performance.csv`)
- **Figure 3**: Evolution of right-leaning bias and text quality index across generations (main experiment)
- **Figure 5**: Alternative setup with center-leaning fine-tuning
- **Figure 6**: Center-leaning article percentage across generations (Appendix B)
- **Figure 7**: Left-leaning article percentage across generations (Appendix B)
- **Figure 8**: Distribution of text quality index across generations (Appendix D)
- **Figure 9**: Average perplexity across generations (Appendix E)
