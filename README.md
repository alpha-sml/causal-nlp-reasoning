# Causal NLP Reasoning

Investigating causal reasoning in natural language using classical machine learning and deep learning approaches on the Corr2Cause benchmark.

## Overview

This project investigates the capability of natural language processing systems to perform causal reasoning by determining whether causal hypotheses logically follow from correlational and conditional independence premises.

- `00_setup.ipynb`: Shared setup notebook for downloading, inspecting, and exporting the Corr2Cause dataset splits into `data/raw/` and `data/processed/`.

## Dataset

This project uses the [Corr2Cause dataset](https://huggingface.co/datasets/causal-nlp/corr2cause), sourced from Hugging Face. It is used to investigate causal reasoning in natural language by evaluating whether causal hypotheses follow from correlational and conditional-independence premises.

Dataset loading, inspection, validation, and export are handled in [`00_setup.ipynb`](00_setup.ipynb). The notebook prepares the data for subsequent feature engineering and model development.

## Directory Structure

- `AML/`: Workflows exploring classical machine learning methods, linguistic feature extraction, causal entity extraction, and syntactic pattern analysis.
- `DL/`: Workflows exploring deep learning methods, tokenization pipelines, neural baselines, and transformer fine-tuning.
- `data/`: Local storage for raw and processed Corr2Cause dataset files.

```text
causal-nlp-reasoning/
├── 00_setup.ipynb
├── AML/
│   ├── AML_01_data_features.ipynb
│   ├── AML_02_ner.ipynb
│   ├── AML_03_dependency_patterns.ipynb
│   ├── AML_04_classical_classifier.ipynb
│   └── AML_05_integration_evaluation.ipynb
├── DL/
│   ├── DL_01_data_pipeline.ipynb
│   ├── DL_02_baseline_models.ipynb
│   ├── DL_03_transformer_models.ipynb
│   ├── DL_04_training_optimization.ipynb
│   └── DL_05_integration_evaluation.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

## Notebooks

The AML and DL notebooks currently contain structural placeholders outlining planned work across each phase.

### Classical Machine Learning (`AML/`)
- `AML_01_data_features.ipynb`: Dataset inspection and feature extraction definition for classical models.
- `AML_02_ner.ipynb`: Variable and entity extraction analysis for abstract causal symbols.
- `AML_03_dependency_patterns.ipynb`: Syntactic dependency and linguistic causal pattern mining.
- `AML_04_classical_classifier.ipynb`: Classical machine learning baseline model training and evaluation.
- `AML_05_integration_evaluation.ipynb`: Classical pipeline integration and final test set evaluation.

### Deep Learning (`DL/`)
- `DL_01_data_pipeline.ipynb`: Dataset tokenization and tensor data pipeline preparation.
- `DL_02_baseline_models.ipynb`: Recurrent and feedforward neural baseline architectures.
- `DL_03_transformer_models.ipynb`: Pretrained transformer fine-tuning for sequence classification.
- `DL_04_training_optimization.ipynb`: Training optimization, regularization, and hyperparameter tuning.
- `DL_05_integration_evaluation.ipynb`: Deep learning integration, cross-model comparison, and final test evaluation.
