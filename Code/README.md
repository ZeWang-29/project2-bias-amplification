# Code

This directory contains all code used in the paper. The experiments were originally run on Kaggle with GPU acceleration. Each script has a **Configuration** section at the top with adjustable file paths and parameters.

## Directory Structure

### `data_preparation/`
Scripts for converting the source dataset (Webis-Bias-Flipper-18) into training-ready text files.

| Script | Description |
|--------|-------------|
| `prepare_mixed_dataset.py` | Samples 506 articles each from Left, Center, and Right categories (1,518 total) from the AllSides CSV. Produces `D_mixed.txt` used as the real dataset in the main experiment. |
| `prepare_center_dataset.py` | Filters center-leaning articles only from the AllSides CSV. Produces `D0.txt` used in the alternative experiment (Appendix I). |

### `training/`
Scripts for iterative fine-tuning across experimental setups.

**Generation 0** (fine-tune GPT-2 on real data):

| Script | Description | Paper Section |
|--------|-------------|---------------|
| `finetune_generation0.py` | Fine-tunes base GPT-2 on `D_mixed.txt` (1,518 real articles). Produces the Generation 0 model (MM1). Uses batch_size=8, lr=5e-5, 5 epochs, weight_decay=0.01. | Section 3.2 |
| `finetune_overfitting_gen0.py` | Fine-tunes GPT-2 with overfitting setup: **25 epochs, weight_decay=0**. | Section 4.3, Overfitting strategy |

**Generation 1** (first synthetic iteration, separate scripts because the initial model source differs):

| Script | Description | Paper Section |
|--------|-------------|---------------|
| `finetune_preservation_gen1.py` | Fine-tunes MM1 on synthetic data DD1 + 152 randomly preserved real articles (10%). | Section 4.3, Preservation |
| `finetune_accumulation_gen1.py` | Fine-tunes MM1 on original data + DD1 combined. | Section 4.3, Accumulation |

**Generation 2+** (full iterative loops with synthetic data generation + fine-tuning):

| Script | Description | Paper Section |
|--------|-------------|---------------|
| `iterative_loop_synthetic.py` | **Main experiment loop.** Loads model from previous generation, generates synthetic articles using 64-token block deterministic decoding (`do_sample=False`), fine-tunes on synthetic data. Repeats for multiple generations. | Section 3.2, 3.3 |
| `iterative_loop_preservation.py` | Same loop but adds 152 randomly selected real articles (10%) to training data each generation. | Section 4.3 |
| `iterative_loop_accumulation.py` | Same loop but accumulates all previous synthetic datasets for training. | Section 4.3 |
| `iterative_loop_overfitting.py` | Same loop with 25 epochs and weight_decay=0. | Section 4.3 |

**Key difference between Gen0 and Gen1+ code**: Gen0 only performs fine-tuning on real data. Gen1+ includes both synthetic data generation (64-token block continuation with deterministic decoding) and fine-tuning in a single loop.

### `evaluation/`
Benchmark scripts for computing the paper's metrics on generated articles.

| Script | Description | Paper Section |
|--------|-------------|---------------|
| `benchmark_classifier.py` | Runs the RoBERTa-base political bias classifier on generated articles. Outputs Left/Center/Right probability scores per article per generation. | Section 3.4 |
| `benchmark_gibberish.py` | Computes the Text Quality Index using the Gibberish Detector. Classifies each sentence as Noise (0), Word Salad (1), Mild Gibberish (2), or Clean (3). | Section 3.5 |
| `benchmark_perplexity.py` | Computes GPT-2 perplexity on generated articles. | Appendix E |

### `mechanistic_analysis/`
Scripts for the neuron-level mechanistic interpretation.

| Script | Description | Paper Section |
|--------|-------------|---------------|
| `extract_activations.py` | Extracts average activation values for all 9,216 neurons (768 per layer x 12 layers) across all 66 fine-tuned GPT-2 models. | Section 3.6 |
| `pearson_correlation.py` | Computes Pearson correlation between neuron weights (or activations) and bias performance (or generation quality) across model versions. | Section 3.6, 4.4 |
| `newey_west_regression.py` | Performs linear regression with Newey-West HAC-adjusted standard errors to test statistical significance of neuron-metric correlations. | Section 3.6, Appendix G |

### `plotting/`
Scripts for reproducing figures in the paper.

| Script | Figure | Description |
|--------|--------|-------------|
| `plot_bias_distribution_fig2.py` | **Figure 2** | Bar chart of Left/Center/Right article distribution for initial GPT-2 outputs |
| `plot_bias_across_generations.py` | **Figure 3a, 5a, 6, 7** | Line plots of bias percentage across generations for different setups |
| `plot_text_quality_index.py` | **Figure 3b, 5b** | Line plots of Text Quality Index across generations with 95% CIs |
| `plot_neuron_weight_bias_correlation.py` | **Figure 4a** | Scatter plot of Pearson correlations between neuron weights and bias performance |
| `plot_neuron_correlation_scatter.py` | **Figure 4** (variant) | Similar scatter plot with layer-colored neurons |
| `plot_pvalue_scatter.py` | **Section 4.4** | Scatter plot of Newey-West adjusted p-values |
| `plot_tqi_distribution.py` | **Figure 8** | Histogram of Text Quality Index distribution across generations |
| `plot_perplexity.py` | **Figure 9** | Average perplexity across generations with 95% CIs |
| `plot_bias_bar_chart.py` | **Figure 2** (variant) | Bar charts of bias distribution per generation |

**Note**: The code for Figure 4b (Neuron Weight vs. Text Quality Index) is analogous to `plot_neuron_weight_bias_correlation.py` — replace the bias performance input CSV with the Text Quality Index correlation CSV.

### `theory/`
Scripts for the theoretical simulation in Appendix L.

| Script | Figure | Description |
|--------|--------|-------------|
| `wmle_simulation.py` | **Figure 10** | Weighted MLE simulation showing bias amplification through iterative estimation with pretrained bias |
| `mle_simulation.py` | **Figure 11** | Standard MLE simulation (control) showing stable estimation without pretrained bias |

## Hyperparameters (Appendix H)

All fine-tuning uses consistent hyperparameters unless stated otherwise:
- Input length: 512 tokens (padded with EOS token)
- Epochs: 5 (25 for Overfitting setup)
- Batch size: 8 (4 for initial Gen0 in some runs)
- Learning rate: 5e-5
- Weight decay: 0.01 (0 for Overfitting setup)
- Synthetic generation: 64-token blocks, deterministic decoding (`do_sample=False`)
