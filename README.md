# Bias Amplification: Large Language Models as Increasingly Biased Media

This repository contains the data and code for the paper:

> **Bias Amplification: Large Language Models as Increasingly Biased Media**
> Ze Wang, Zekun Wu, Yichi Zhang, Xin Guan, Navya Jain, Qinyang Lu, Saloni Gupta, Adriano Koshiyama
> *Proceedings of the 14th International Joint Conference on Natural Language Processing (IJCNLP) and the 4th Conference of the Asia-Pacific Chapter of the ACL*, 2025.
> [Paper](https://aclanthology.org/2025.ijcnlp-long.8/)

## Overview

We investigate political bias amplification in GPT-2 through iterative synthetic fine-tuning. Our experiments show that GPT-2 progressively exhibits stronger right-leaning bias over successive training generations, even when starting from an unbiased dataset. We evaluate three mitigation strategies (Overfitting, Preservation, Accumulation) and propose a mechanistic analysis identifying distinct neuron populations driving bias amplification and model collapse.

## Repository Structure

```
.
├── Data/
│   ├── Bias_Performance_and_Generation_Quality/   # Classifier scores, TQI, perplexity (Figs 2,3,5-9)
│   └── Mechanistic_Interpretation/                # Neuron correlations, regression results (Fig 4, Sec 4.4)
├── Code/
│   ├── data_preparation/      # Dataset preparation scripts
│   ├── training/              # Iterative fine-tuning and synthetic data generation
│   ├── evaluation/            # Bias classifier, Gibberish Detector, perplexity benchmarks
│   ├── mechanistic_analysis/  # Neuron activation extraction, correlation, Newey-West tests
│   ├── plotting/              # Scripts to reproduce all paper figures
│   └── theory/                # WMLE/MLE simulations (Appendix L)
├── LICENSE
└── README.md
```

## Data

The `Data/` directory contains all processed experiment results needed to reproduce every figure in the paper. See [`Data/Bias_Performance_and_Generation_Quality/README.md`](Data/Bias_Performance_and_Generation_Quality/README.md) and [`Data/Mechanistic_Interpretation/README.md`](Data/Mechanistic_Interpretation/README.md) for details.

**Note**: Generation numbering in the CSV files is offset by 1 (Generation 1 in CSVs = Generation 0 in the paper).

## Code

The `Code/` directory contains all scripts used in the paper, organized by function. See [`Code/README.md`](Code/README.md) for a complete mapping of scripts to paper figures/sections.

The experiments were originally run on Kaggle with GPU acceleration. Each script has a **Configuration** section at the top with adjustable file paths and parameters. Plotting scripts reference the `Data/` directory directly and can reproduce figures without additional setup.

### Key Components

- **Iterative Fine-tuning Pipeline**: GPT-2 is fine-tuned on real news articles (Generation 0), then iteratively on synthetic articles generated via 64-token block deterministic decoding (Generations 1-10).
- **Political Bias Classifier**: A RoBERTa-base model (macro F1 = 0.9196) trained on the Webis-Bias-Flipper-18 dataset to classify articles as Left, Center, or Right.
- **Text Quality Index**: Based on the [Gibberish Detector](https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457), scoring sentences from 0 (Noise) to 3 (Clean).
- **Mechanistic Analysis**: Pearson correlations and Newey-West regressions across 9,216 neurons and 66 model versions.

## Models

Fine-tuned GPT-2 models are available on HuggingFace under the `refipsai` organization:
- `refipsai/MM1` - `refipsai/MM11`: Main experiment (Synthetic baseline)
- `refipsai/MMP1` - `refipsai/MMP11`: Preservation (10% real data)
- `refipsai/MMA1` - `refipsai/MMA11`: Accumulation
- `refipsai/MMO1` - `refipsai/MMO11`: Overfitting

## Source Dataset

The original news articles are from the [Webis-Bias-Flipper-18](https://zenodo.org/records/3271061) dataset (Chen et al., 2018).

## Citation

```bibtex
@inproceedings{wang-etal-2025-bias,
    title = "Bias Amplification: Large Language Models as Increasingly Biased Media",
    author = "Wang, Ze  and
      Wu, Zekun  and
      Zhang, Yichi  and
      Guan, Xin  and
      Jain, Navya  and
      Lu, Qinyang  and
      Gupta, Saloni  and
      Koshiyama, Adriano",
    editor = "Inui, Kentaro  and
      Sakti, Sakriani  and
      Wang, Haofen  and
      Wong, Derek F.  and
      Bhattacharyya, Pushpak  and
      Banerjee, Biplab  and
      Ekbal, Asif  and
      Chakraborty, Tanmoy  and
      Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.8/",
    pages = "115--132",
    ISBN = "979-8-89176-298-5",
    abstract = "Model collapse{---}a phenomenon where models degrade in performance due to indiscriminate use of synthetic data{---}is well studied. However, its role in bias amplification{---}the progressive reinforcement of pre-existing social biases in Large Language Models (LLMs){---}remains underexplored. In this paper, we formally define the conditions for bias amplification and demonstrate through statistical simulations that bias can intensify even in the absence of sampling errors, the primary driver of model collapse. Empirically, we investigate political bias amplification in GPT-2 using a custom-built benchmark for sentence continuation tasks. Our findings reveal a progressively increasing right-leaning bias. Furthermore, we evaluate three mitigation strategies{---}Overfitting, Preservation, and Accumulation{---}and show that bias amplification persists even when model collapse is mitigated. Finally, a mechanistic interpretation identifies distinct sets of neurons responsible for model collapse and bias amplification, suggesting they arise from different underlying mechanisms."
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
