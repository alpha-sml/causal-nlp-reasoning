# Causal NLP Reasoning

Investigating causal reasoning in natural language using classical machine learning and deep learning approaches on the Corr2Cause benchmark.

## Overview

This project investigates whether natural language processing systems can perform rigorous causal reasoning: specifically, determining whether a natural language causal hypothesis logically follows from premises specifying statistical correlations and conditional independencies.

The benchmark dataset used is **[Corr2Cause](https://huggingface.co/datasets/causal-nlp/corr2cause)** (Jin et al.), an NLI-formulated causal reasoning dataset where each instance consists of:
- **Premise**: Natural language sentences defining a closed variable system (train has 4–6-variable examples, while validation and test include 2–6-variable examples) and reporting observed correlations and conditional independencies.
- **Hypothesis**: A causal relation query asserting direct causation, indirect causation, common confounding, or absence of causation.
- **Label**: Binary classification (`1` if the causal claim is valid and logically entailed by the observational premises under causal Markov and faithfulness assumptions, `0` otherwise).

---

## Repository Structure & Notebook Pipeline

```text
causal-nlp-reasoning/
├── 00_setup.ipynb                      # Data acquisition, split validation, and canonical export
├── 01_eda.ipynb                        # Exploratory data analysis, distributions, and template profiling
├── AML/
│   ├── AML_01_data_features.ipynb      # Linguistic and structural feature engineering (19 features)
│   ├── AML_02_ner.ipynb                # Symbolic entity & variable extraction with character offsets
│   ├── AML_03_dependency_patterns.ipynb# Syntactic & rule-based causal constraint extraction (in-memory)
│   ├── AML_04_classical_classifier.ipynb# [Planned] Classical ML classification baselines
│   └── AML_05_integration_evaluation.ipynb# [Planned] Neuro-symbolic causal pipeline & graph evaluation
├── DL/                                 # [Planned] Deep learning & transformer fine-tuning workflows
│   ├── DL_01_data_pipeline.ipynb
│   ├── DL_02_baseline_models.ipynb
│   ├── DL_03_transformer_models.ipynb
│   ├── DL_04_training_optimization.ipynb
│   └── DL_05_integration_evaluation.ipynb
├── data/
│   ├── raw/                            # Benchmark splits (train.csv, validation.csv, test.csv)
│   └── processed/                      # Canonical splits and feature-enriched artifacts
├── requirements.txt                    # Project dependencies with minimum-version specifications
└── README.md
```

### Notebook Modules & Implementation Details

#### In-Scope Notebooks:
- **`00_setup.ipynb`**:
  - Automatically locates the repository root and initializes directory structures.
  - Downloads the official Corr2Cause splits directly from the Hugging Face Hub (`causal-nlp/corr2cause`).
  - By default (`FORCE_REBUILD = False`), reuses valid pre-existing datasets without rewriting. Setting `FORCE_REBUILD = True` triggers an explicit fresh rebuild targeting only the five canonical CSV files.
  - Ensures downstream processed files are regenerated from freshly exported raw splits.
  - Exports validated splits to `data/raw/` (`train.csv`, `validation.csv`, `test.csv`) and `data/processed/` (`train_val.csv`, `test.csv`).
  - Enforces row-count and schema validation: 205,734 train, 1,076 validation, 1,162 test records.

- **`01_eda.ipynb`**:
  - Comprehensive exploratory data analysis across variable counts (train has 4–6-variable examples, while validation and test include 2–6-variable examples) and causal relation templates.
  - Accurately characterizes severe target class imbalance across all splits:
    - **Train**: 81.51% Refuted (0) vs. 18.49% Entailed (1)
    - **Validation**: 87.08% Refuted (0) vs. 12.92% Entailed (1)
    - **Test**: 84.51% Refuted (0) vs. 15.49% Entailed (1)
  - Evaluates sentence lengths, vocabulary distributions, and conditioning set complexities.
  - Evaluates schema consistency, template validity, variable validity, and cross-split partition isolation.
  - Zero machine-specific path dependencies.

- **`AML/AML_01_data_features.ipynb`**:
  - Extracts 19 engineered linguistic, structural, ratio, and variable interaction features without target leakage across 5 categories:
    - Text complexity / length metrics (7): `input_char_count`, `input_word_count`, `premise_char_count`, `premise_word_count`, `premise_sentence_count`, `hypothesis_char_count`, `hypothesis_word_count`.
    - Statistical premise clause metrics (4): `n_correlations`, `n_independences`, `n_given_clauses`, `has_however`.
    - Ratios & relational aggregations (3): `premise_hyp_word_ratio`, `indep_corr_ratio`, `n_total_relations`.
    - Hypothesis causal intent (1): `hyp_causal_keyword_count`.
    - Symbolic variable interactions (4): `n_premise_variables`, `n_hypothesis_variables`, `variable_overlap`, `all_hyp_vars_in_premise`.
  - Maintains strict train/test isolation.
  - Exports `data/processed/train_val_features.csv` (206,810 rows × 26 columns) and `data/processed/test_features.csv` (1,162 rows × 25 columns).

- **`AML/AML_02_ner.ipynb`**:
  - Extracts abstract symbolic variables (e.g., `A`, `B`, `C`) with exact character offsets (`start`, `end`, `label="VARIABLE"`).
  - Implements disambiguation for English indefinite article "A" vs. causal variable "A".
  - Evaluates comparative performance between symbolic extraction and off-the-shelf general-purpose NER models (with graceful fallback when spaCy is unavailable).
  - Exports `data/processed/ner_train_val.csv` and `data/processed/ner_test.csv` containing normalized variable sets, entity offsets, and overlap features.

- **`AML/AML_03_dependency_patterns.ipynb`**:
  - Mines syntactic and linguistic patterns to extract candidate relations (direct causation, indirect ancestral paths, negated causation, confounders, and colliders).
  - Implements hybrid dependency-parser extraction with deterministic rule-fallback grammar.
  - Maps natural language premise statements and causal hypothesis queries into structured constraint representations (`directed_edges`, `directed_paths`, `forbidden_edges`, `conditional_independencies`, `common_causes`) designed for Bayesian network reasoning in `AML_05`.
  - Operates strictly in-memory without untracked disk artifacts.

#### Planned / Future Modules (Placeholders):
- **`AML/AML_04_classical_classifier.ipynb`**: Classical ML classifiers (Logistic Regression, Random Forest, XGBoost/LightGBM) trained on engineered tabular and lexical features.
- **`AML/AML_05_integration_evaluation.ipynb`**: Neuro-symbolic integration combining statistical classifiers with exact causal DAG constraint satisfaction.
- **`DL/DL_01` through `DL_05`**: Deep learning workflows covering neural baselines (BiLSTM), pretrained transformer fine-tuning (BERT, RoBERTa, DeBERTa), optimization, and evaluation.

---

## Environment Setup & Dependency Compatibility

The project targets **Python 3.14** (tested and verified on Python 3.14.2 macOS arm64). All active data preparation, exploratory analysis, and feature engineering workflows (`00_setup.ipynb`, `01_eda.ipynb`, and `AML_01` through `AML_03`) as well as planned downstream dependencies (`torch`, `pgmpy`, `transformers`, `scikit-learn`) have been audited against Python 3.14.

### Quickstart

```bash
# 1. Create and activate a Python 3.14 virtual environment
# Note: If the workspace path contains a colon ':' (e.g. 'AML:DL Project'),
# create the virtualenv outside the path to prevent venv path separator collisions:
python3.14 -m venv ~/.venvs/causal-nlp-314
source ~/.venvs/causal-nlp-314/bin/activate

# 2. Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# 3. Install declared dependencies
pip install -r requirements.txt

# 4. Download spaCy English model for NLP dependency extraction
python -m spacy download en_core_web_sm

# 5. Register Jupyter kernel for notebook execution
python -m ipykernel install --user --name python3 --display-name "Python 3.14 (causal-nlp)"
```

### Dependency Classification & Verification Matrix

The table below delineates the dependencies declared in `requirements.txt`, the modules that actively execute them, the versions verified in the execution environment (`~/.venvs/causal-nlp-314`), and the versions resolved during clean installation in an isolated fresh Python 3.14.2 environment (`scratch/fresh_venv`):

| Package | Declared in `requirements.txt` | Pipeline Role | Tested Active Version | Clean Fresh Venv Version | Release Type / Binary Wheel Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`pandas`** | `>=2.2.0` | Active (`00_setup`, `01_eda`, `AML_01`–`03`) | `3.0.1` | `3.0.6` | PyPI standard release binary wheel |
| **`numpy`** | `>=1.26.0` | Active (`01_eda`, `AML_01`–`03`) | `2.4.2` | `2.4.2` | PyPI standard release binary wheel |
| **`datasets`** | `>=2.19.0` | Active (`00_setup`) | `5.1.0` | `5.1.0` | PyPI standard release wheel |
| **`networkx`** | `>=3.0` | Active (`AML_03`) | `3.7` | `3.7` | PyPI standard release wheel |
| **`matplotlib`** | `>=3.8.0` | Active (`01_eda`) | `3.11.1` | `3.11.2` | PyPI standard release binary wheel |
| **`seaborn`** | `>=0.13.0` | Active (`01_eda`) | `0.13.2` | `0.13.2` | PyPI standard release wheel |
| **`scipy`** | `>=1.12.0` | Active (`01_eda`, `AML_01`) | `1.18.1` | `1.18.1` | PyPI standard release binary wheel |
| **`ipykernel`** | `>=6.29.0` | Active (Jupyter notebook execution) | `7.2.0` | `7.4.0` | PyPI standard release binary wheel |
| **`spacy`** | `>=3.7.0` | Active (`AML_02`, `AML_03`) | `3.8.16` | `3.8.16` | PyPI standard release binary wheel (`en_core_web_sm 3.8.0` verified; fallback regex available) |
| **`scikit-learn`** | `>=1.4.0` | Active (`01_eda` shortcut diagnostics) & Planned (`AML_04`) | `1.9.0` | `1.9.1` | PyPI standard release binary wheel |
| **`transformers`** | `>=4.40.0` | Planned (`DL/DL_03_transformer_models.ipynb`) | `5.20.2` | `5.20.2` | PyPI standard release wheel |
| **`torch`** | `>=2.2.0` | Planned (`DL/DL_01`–`05` deep learning) | `2.14.1` | `2.14.1` | PyPI standard release binary wheel (`cp314-cp314-macosx_14_0_arm64`, MPS verified) |
| **`pgmpy`** | `>=0.1.25` | Planned (`AML_05_integration_evaluation.ipynb`)| `1.1.2` | `1.1.2` | PyPI standard release wheel |

### Detailed Dependency Notes & Platform Verification Scope

1. **Active vs. Planned Module Separation**:
   - **Active Dependencies**: The five currently implemented and executed notebooks (`00_setup.ipynb`, `01_eda.ipynb`, `AML_01_data_features.ipynb`, `AML_02_ner.ipynb`, and `AML_03_dependency_patterns.ipynb`) actively import and execute: `pandas`, `numpy`, `datasets`, `networkx`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn` (used in `01_eda` for diagnostic shortcut baselines: DummyClassifier and DecisionTreeClassifier), `ipykernel`, and `spacy` (with deterministic fallback).
   - **Planned Dependencies**: The heavier ML/DL dependencies—`torch`, `transformers`, and `pgmpy` (as well as extended `scikit-learn` estimators in `AML_04`)—are declared in `requirements.txt` to prepare for subsequent modules (`AML_04`, `AML_05`, and `DL/`). While they install cleanly and import without errors, they are **not** invoked in the currently active data preparation and feature extraction notebooks.
2. **Release Status & Pre-Release Packages**:
   - All 13 primary packages declared in `requirements.txt` are official, standard releases available from PyPI. None of the primary dependencies require experimental, development, or nightly (`dev`) builds on Python 3.14.
   - Across the transitive dependency graph resolved during clean installation, exactly one indirect package was installed as a release candidate: `wrapt==2.5.2rc1` (an indirect C-extension dependency required by `smart-open` for `weasel`/`spacy`).
3. **Fresh Environment Verification Evidence**:
   - A clean, isolated Python 3.14.2 virtual environment was created from scratch without inherited packages (`scratch/fresh_venv`).
   - `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm` completed with exit code 0 and zero compilation failures.
   - Comprehensive smoke tests verified: tensor allocation and operations on Apple Silicon (MPS) with `torch 2.14.1`, graph edge construction with `networkx 3.7`, feature dataframe operations with `pandas 3.0.6`, model inference with `spacy 3.8.16`, and probabilistic network imports with `pgmpy 1.1.2`.
4. **Scope of Portability Claims**:
   - Direct verification has been performed on **macOS arm64 (Apple Silicon) under Python 3.14.2**.
   - Because PyPI binary wheel availability for newly released Python versions can differ across operating systems and CPU architectures (e.g. Windows x86_64 or Linux glibc variations), compatibility is guaranteed only for verified macOS arm64 environments. Environments without prebuilt wheels may require local C/C++ compilation tools.
5. **Python 3.15+**:
   - Python 3.15 is outside the target runtime and is neither tested nor supported.

---

## Dataset Artifacts & Data Directory Layout

Running the notebooks in sequence regenerates all processed artifacts:

| Path | Description | Shape | Size (Decimal MB / Binary MiB) |
| :--- | :--- | :--- | :--- |
| `data/raw/train.csv` | Official training split from Hugging Face | 205,734 × 4 | 383.42 MB (365.66 MiB) |
| `data/raw/validation.csv` | Official validation split from Hugging Face | 1,076 × 4 | 1.46 MB (1.39 MiB) |
| `data/raw/test.csv` | Official held-out test split from Hugging Face | 1,162 × 4 | 1.18 MB (1.13 MiB) |
| `data/processed/train_val.csv` | Combined training + validation pool | 206,810 × 4 | 384.88 MB (367.05 MiB) |
| `data/processed/test.csv` | Canonical held-out test split | 1,162 × 4 | 1.18 MB (1.13 MiB) |
| `data/processed/train_val_features.csv` | 19 engineered features across 5 categories (AML_01) | 206,810 × 26 | 775.73 MB (739.79 MiB) |
| `data/processed/test_features.csv` | 19 engineered features for test set (AML_01) | 1,162 × 25 | 2.38 MB (2.27 MiB) |
| `data/processed/ner_train_val.csv` | Extracted variables, entity offsets, and overlap features (AML_02) | 206,810 × 14 | 3,224.41 MB (3,075.04 MiB) |
| `data/processed/ner_test.csv` | Extracted variables, entity offsets, and overlap features for test set (AML_02) | 1,162 × 13 | 8.80 MB (8.39 MiB) |

*Note: Sizes are measured directly from the generated CSV artifacts in the workspace (decimal MB where $1\text{ MB} = 10^6\text{ bytes}$; binary MiB where $1\text{ MiB} = 2^{20}\text{ bytes}$).*

### CSV Record Count vs. Physical Line Count (`wc -l`)

* **Quoted Multiline Fields**: In the raw and feature CSV files (`train.csv`, `validation.csv`, `test.csv`, `train_val.csv`, `train_val_features.csv`), the `input` field contains an embedded newline separator (`\n`) between the premise statement and the hypothesis query (`Premise: ...\nHypothesis: ...`).
* **Physical Lines (`wc -l`)**: Because each data record spans exactly two physical text lines, `wc -l` reports $2N + 1$ lines ($411,469$ lines for train; $2,153$ for validation; $2,325$ for test; $413,621$ for combined train+val).
* **Logical DataFrame Records**: Standard RFC 4180 CSV parsers (including `pandas.read_csv()`) correctly parse multiline quoted text fields, yielding the exact ground-truth record counts: **205,734 train**, **1,076 validation**, **1,162 test**, and **206,810 combined train+val**.
* **NER Artifacts**: In `ner_train_val.csv` and `ner_test.csv`, `input` is omitted to reduce redundancy (since `premise` and `hypothesis` are preserved as distinct fields without embedded newlines), so `wc -l` on NER files matches $N + 1$ physical lines ($206,811$ and $1,163$).

### Extraction Limitations & Linguistic Nuances

1. **Occurrence-Level Article vs. Variable Disambiguation & Substring Verification**:
   - In Corr2Cause, variable `A` is distinguished from English indefinite article `a`/`A` based on local grammatical context.
   - Punctuation (`A,`, `(A)`), post-prepositional slots (`of A`, `between A and B`), and subject/verb slots (`A causes B`, `A is a cause of B`) signify variable references.
   - Capitalized sentence-initial tokens followed by common noun/adjective phrases (`A closed system`, `A direct cause`, `A path exists`) represent indefinite articles.
   - All extracted entity character offsets (`start`, `end`) are strictly validated to prove `source_text[start:end] == entity_text` with 100% agreement across all benchmark and regression suites.
2. **Candidate Relation Extraction vs. Causal Graph Satisfiability**:
   - `AML_03` extracts structural candidate relations (e.g. direct causes, indirect paths, conditional independencies).
   - High candidate recall indicates syntactic pattern coverage across text premises, but does **not** equal mathematical proof of causal graph entailment. Entailment requires solving the full constraint satisfaction problem (d-separation and Markov equivalence classes) implemented in downstream modules (`AML_05`).
3. **Exact Relation Phrase Evidence Spans**:
   - In `AML_03`, candidate relations (both from the spaCy dependency parser and deterministic fallback grammar) isolate the exact source phrase supporting the relation (e.g. `A causes B`, `B is caused by A`, `confounder of A and B`, `A and B are independent given C`).
   - Offsets refer strictly to the exact source text (`source_text[start_char:end_char] == evidence_text`) without whole-text assignment or normalization offset drift. An explicit normalization mapper (`map_normalized_span_to_source`) guarantees exact source boundary recovery when matching against normalized text.

### Reproduction Sequence

To reproduce the analysis and regenerate all data:

1. **`00_setup.ipynb`**: Fetch benchmark data from Hugging Face Hub and write raw/processed partitions. By default (`FORCE_REBUILD = False`), existing valid files are safely verified and reused without deletion. Setting `FORCE_REBUILD = True` triggers a safe, fresh re-export of only the five canonical CSV files.
2. **`01_eda.ipynb`**: Verify distributions and benchmark properties across Train, Validation, and Test partitions.
3. **`AML/AML_01_data_features.ipynb`**: Compute 19 linguistic and structural features and export feature tables.
4. **`AML/AML_02_ner.ipynb`**: Extract symbolic variable entities and character offsets.
5. **`AML/AML_03_dependency_patterns.ipynb`**: Parse syntactic dependencies and build causal graph constraints (strictly in-memory).
