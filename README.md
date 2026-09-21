# Causal NLP Reasoning

Project structure for extracting causal relationships from natural-language documents and building a neuro-symbolic causal graph.

## Structure

```text
causal-nlp-reasoning/
├── data/
│   ├── raw/                       # Store the Corr2Cause dataset or chosen alternative
│   └── processed/                 # Documents split before extraction to prevent text leakage
├── 01_feature_engineering.ipynb   # Entity schema definition, alias deduplication, and data splitting
├── 02_aml_baseline.ipynb          # Dependency patterns, NER, and lightweight relation classifier
├── 03_dl_pipeline.ipynb           # Fine-tuning DistilBERT or instruction model for pair labeling
├── 04_neuro_symbolic.ipynb        # Construct DAG, reject self-loops, and fit pgmpy Bayesian network
├── 05_final_submission.ipynb      # Reproducible notebook visualizing the DAG, comparing methods, and testing queries
├── requirements.txt               # pandas, scikit-learn, transformers, torch, spacy, networkx, pgmpy
└── README.md
```
