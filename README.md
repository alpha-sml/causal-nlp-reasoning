# Causal NLP Reasoning

Investigating causal relation extraction from scientific text, DAG construction, and Bayesian network inference alongside synthetic causal reasoning benchmarks.

## Overview

In alignment with **`DL_AML_Project_Idea.pdf`** (*Project 1: Causal NLP Reasoning — Neuro-Symbolic Causal Inference via Language Model Knowledge Extraction and Bayesian Networks*), this repository implements an end-to-end causal natural language processing pipeline.

The repository separates the project into two distinct tasks:

1. **Primary Task: Biomedical Causal Relation Extraction, DAG Construction, and Probabilistic Modeling (BioCause)**
   - **Corpus:** The [BioCause](https://www.nactem.ac.uk/biocause/) corpus developed by the National Centre for Text Mining (NaCTeM) at the University of Manchester ([Mihăilă et al., 2013](https://doi.org/10.1186/1471-2105-14-2)), built upon the BioNLP 2011 Shared Task Infectious Diseases (ID) dataset.
   - **Data Structure:** 20 full-text open-access articles from PubMed Central across 198 paired section files (`.txt` and `.ann`), containing 851 gold causal relation events (+ 1 branching event yielding 852 directed relation triples), 1,526 causal argument spans, and 381 distinct causal trigger expressions.
   - **Licensing & Terms:** Full-text articles are from the PMC Open Access subset. Causal annotations are copyrighted by NaCTeM and licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-nc-sa/3.0/) (CC BY-NC-SA 3.0). BioCause files and derived texts are strictly stored in local git-ignored directories (`data/raw/` and `data/processed/`) and are never committed to version control.
   - **Article-Level Group Splitting:** To prevent text and domain leakage, splitting is performed strictly by source publication (`article_id`): 12 articles for training, 4 for validation, and 4 for held-out testing.
   - **Graph Construction & Causal Inference:** Validated causal extractions are compiled into directed NetworkX graphs with stable identifier keys `(section_id, annotation_id)`, audited for structural invariants (self-loops, cycles, conflicting directions), and modeled probabilistically in `pgmpy` to evaluate observational associations versus Pearl's interventional queries ($do$-calculus via graph surgery).

2. **Supplementary Task: Abstract Causal Entailment Reasoning (Corr2Cause)**
   - Preserves the [Corr2Cause](https://huggingface.co/datasets/causal-nlp/corr2cause) benchmark ([Jin et al., 2024](https://arxiv.org/abs/2306.05836)) as a supplementary causal reasoning experiment.
   - Tests whether formal causal claims follow logically from statistical correlations and conditional independencies under causal Markov and faithfulness assumptions.
   - **Boundary Definition:** Corr2Cause contains binary entailment labels on synthetic abstract statements; it does **not** provide character-level relation extraction spans or gold biomedical triples. Its metrics are strictly isolated from BioCause relation extraction.

---

## Repository Structure & Notebook Pipeline

```text
causal-nlp-reasoning/
├── 00_setup.ipynb                      # Centralized entry point: BioCause corpus acquisition & validation, Corr2Cause setup
├── 01_eda.ipynb                        # Exploratory data analysis, BioCause standoff profiling, and template distributions
├── AML/
│   ├── AML_01_data_features.ipynb      # BioCause standoff parsing, 100% offset verification, article group splitting & candidate generation
│   ├── AML_02_ner.ipynb                # Scientific entity & argument mention extraction, conservative canonicalization & spaCy baseline evaluation
│   ├── AML_03_dependency_patterns.ipynb# Syntactic dependency parsing & lexical pattern causal extraction with directional correctness
│   ├── AML_04_classical_classifier.ipynb# Supervised relation baseline, stable-ID NetworkX DAG construction & pgmpy Bayesian network reasoning
│   ├── AML_05_integration_evaluation.ipynb# End-to-end multi-pathway benchmark, unified schema, DAG evaluation & error analysis
│   └── causal_extractors.py            # Shared direction-aware lexical and syntactic dependency relation extractors
├── DL/                                 # [Planned] Deep learning & transformer fine-tuning workflows
│   ├── DL_01_data_pipeline.ipynb
│   ├── DL_02_baseline_models.ipynb
│   ├── DL_03_transformer_models.ipynb
│   ├── DL_04_training_optimization.ipynb
│   └── DL_05_integration_evaluation.ipynb
├── data/
│   ├── raw/                            # Ignored raw corpora (biocause.zip, biocause/, train.csv, validation.csv, test.csv)
│   └── processed/                      # Ignored standardized CSV tables and candidate pairs
├── requirements.txt                    # Active pipeline dependencies (Python 3.14.2 macOS arm64)
├── requirements-dl.txt                 # Optional deep learning dependencies (torch, transformers, accelerate)
└── README.md
```

### Notebook Modules & Implementation Details

#### Active Pipeline Notebooks (Executed & Verified Sequentially from Fresh Kernels):

- **`00_setup.ipynb`**:
  - Centralized setup entry point for acquiring and preparing both project datasets.
  - **BioCause Acquisition & Structural Validation:**
    - Downloads official BioCause archive safely to a uniquely named temporary file from NaCTeM distribution URL (`https://www.nactem.ac.uk/biocause/download-biocause-corpus.php`).
    - Validates zip archive integrity with `testzip()` and checks against zip-slip path traversal vulnerabilities.
    - Extracts into an isolated temporary directory before validating paired files.
    - Validates that all 198 section `.txt` and `.ann` files are present and paired across 20 PMC articles.
    - Confirms `LICENSE.txt` (CC BY-NC-SA 3.0) and verifies presence of 12,939 annotations and 851 causal events with 0 broken references.
    - Promotes validated corpus atomically to `data/raw/biocause/BioCause_corpus`.
    - Implements safe reuse mode: skips downloading when a valid local extracted corpus already exists.
  - **Corr2Cause Ingestion & Export:**
    - Validates local raw splits (`train.csv`: 205,734, `validation.csv`: 1,076, `test.csv`: 1,162).
    - Rebuilds and verifies `data/processed/train_val.csv` (206,810 rows) and `data/processed/test.csv` (1,162 rows) via transactional staging and SHA-256 integrity verification.

- **`01_eda.ipynb`**:
  - Comprehensive exploratory data analysis across partitions.
  - **BioCause Standoff Profiling:** Profiles 198 section files, 20 PMC articles, annotation structure, and validates candidate split isolation (12 train, 4 val, 4 test articles strictly disjoint).
  - Inspects file dimensions, schemas, class balance, and text length distributions.
  - Analyzes structural templates (direct cause, collider, confounder, mediator) across variable counts (4, 5, 6).

- **`AML/AML_01_data_features.ipynb`**:
  - Consumes the validated BioCause corpus prepared by `00_setup.ipynb` (does not maintain an independent download implementation).
  - Parses stand-off `.ann` and `.txt` files; verifies 100% of character offsets with zero mismatches against raw source texts.
  - Invariant verified: 0 discontinuous spans exist in BioCause; parser asserts this corpus invariant.
  - Handles branching causal events (E36 in PMC2242835-02-Results-06 with Cause, Effect, Effect2 yielding 2 relation triples, resulting in 852 total gold relations).
  - Enforces deterministic, isolated group splitting by `article_id` (12 train, 4 validation, 4 test; seed 42) with strict disjointness assertions (`train ∩ val == ∅`, `train ∩ test == ∅`, `val ∩ test == ∅`).
  - Generates candidate argument pairs for relation extraction: all ordered pairs of argument spans (`Span` or `Fact`) within a 2,000-character window in the same section file.
    - **Window Coverage:** 852 out of 852 gold relations (100.00%) fall within the 2,000-character window (0 fall outside).
  - Exports standardized tables to `data/processed/`:
    - `biocause_relations.csv`: 852 gold standard causal relation triples with exact character offsets.
    - `biocause_entities.csv`: 12,939 gold annotated entity and situation mentions.
    - `biocause_candidates.csv`: 13,408 candidate argument pairs (6.35% causal rate).
  - Preserves Corr2Cause premise/hypothesis extraction and engineers 19 linguistic/structural features without target leakage; exports `train_val_features.csv` (206,810 rows) and `test_features.csv` (1,162 rows).

- **`AML/AML_02_ner.ipynb`**:
  - Implements scientific entity and causal argument mention extraction.
  - Distinguishes ground truth gold references (`is_gold=1`) from model predictions (`is_gold=0`).
  - Establishes stable mention contracts (`mention_id`, `doc_id`, `start_char`, `end_char`, `text`, `label`, `canonical_id`, `is_gold`).
  - Implements conservative alias normalization mapping established biological synonyms (e.g., `pmra` -> `PMRA`, `polymyxin b` -> `POLYMYXIN_B`) while strictly avoiding improper edit-distance merging of distinct regulators (`PmrA` vs `PmrB`).
  - Evaluates off-the-shelf spaCy `en_core_web_sm` under three explicit regimes:
    1. **Span-Only Exact Matching** (Boundary-only exact offset equality):
       - Conventional Biomedical Entities: Validation F1 = **20.91%** (P = 26.50%, R = 17.27%); Test F1 = **13.43%** (P = 15.98%, R = 11.59%)
       - All BioCause Spans: Validation F1 = **15.39%** (P = 26.72%, R = 10.81%); Test F1 = **10.66%** (P = 15.98%, R = 7.99%)
    2. **Typed Exact Matching** (Offsets + label compatibility under defined mapping policy):
       - Conventional Biomedical Entities: Validation F1 = **17.67%** (P = 22.39%, R = 14.60%); Test F1 = **8.70%** (P = 10.35%, R = 7.51%)
       - All BioCause Spans: Validation F1 = **12.90%** (P = 22.39%, R = 9.06%); Test F1 = **6.90%** (P = 10.35%, R = 5.18%)
    3. **Deterministic Maximum Relaxed Matching** (Hungarian algorithm maximizing IoU with $\ge 0.5$ threshold):
       - Conventional Biomedical Entities: Validation F1 = **23.01%** (P = 29.16%, R = 19.00%); Test F1 = **15.38%** (P = 18.29%, R = 13.27%)
       - All BioCause Spans: Validation F1 = **17.11%** (P = 29.71%, R = 12.02%); Test F1 = **12.20%** (P = 18.29%, R = 9.15%)
  - Explains the ontology mismatch and domain gap between web NER models and specialized biomedical situation spans.
  - Preserves symbolic single-letter variable extraction unit tests for Corr2Cause.

- **`AML/AML_03_dependency_patterns.ipynb`**:
  - Implements both deterministic lexical patterns and spaCy syntactic dependency parse tree extraction with strict **directional correctness**:
    - **Active Causal Rules:** Direct promotion, causation, induction (`cause`, `induce`, `activate`, `promote`, `lead to`, `results in`).
    - **Passive / Inverted Constructions:** Normalized to canonical `(cause -> effect)` (`results from`, `is caused by`, `is due to`, `is mediated by`).
    - **Directional Correctness Assertion:** A prediction is evaluated as positive for directed candidate pair `(arg1, arg2)` only when the predicted cause is `arg1` and the predicted effect is `arg2`. Inverted predictions are rejected as non-matches for that edge direction.
    - **Negation Filtering:** Suppresses negated assertions (`does not cause`, `failed to induce`).
    - **Epistemic Modality:** Attaches uncertainty confidence (`may`, `might`, `suggests`, `possible` -> confidence = 0.70).
    - **Non-Causal Controls:** Rejects mere statistical correlation (`correlated with`, `associated with`) and self-loops.
  - 10-case unit test suite verifying forward order, reversed passive order, negation rejection, correlation rejection, uncertainty detection, and self-loop suppression (10/10 passed).
  - Empirical evaluation on candidate pairs using stable candidate IDs across document-held-out splits:
    - **Lexical Rule Baseline:**
      - Validation: Precision = **39.09%**, Recall = **32.82%**, F1 = **35.68%**, Coverage = **5.75%**
      - Test: Precision = **40.74%**, Recall = **33.00%**, F1 = **36.46%**, Coverage = **5.75%**
    - **spaCy Dependency Extractor:**
      - Validation: Precision = **23.42%**, Recall = **19.85%**, F1 = **21.49%**, Coverage = **5.80%**
      - Test: Precision = **34.72%**, Recall = **25.00%**, F1 = **29.07%**, Coverage = **5.11%**
    - **Hybrid Extractor:**
      - Validation: Precision = **31.38%**, Recall = **45.04%**, F1 = **36.99%**, Coverage = **9.82%**
      - Test: Precision = **37.93%**, Recall = **44.00%**, F1 = **40.74%**, Coverage = **8.24%**
  - Preserves Corr2Cause constraint parsing for Bayesian network translation.

- **`AML/AML_04_classical_classifier.ipynb`**:
  - **Supervised Relation Classification on BioCause Candidates:**
    - Task definition: Relation classification conditional on gold argument spans (classifying whether candidate argument pairs represent a true causal relation).
    - Constructs dense positional/structural features, lexical trigger indicators, and TF-IDF n-grams fitted strictly on `X_train`.
    - Evaluates Majority Dummy, unweighted Logistic Regression, class-balanced Logistic Regression, and class-balanced LinearSVC under severe class imbalance (~6.3% positive).
    - Selects `Logistic Regression (Unweighted)` based strictly on **Validation Macro-F1** (**0.7412**) and Causal F1 (**0.5079**).
    - Single frozen Held-Out Test Evaluation (1,408 test examples): **Accuracy = 94.18%**, **Macro-F1 = 71.13%**, **Causal Precision = 68.00%**, **Causal Recall = 34.00%**, **Causal F1 = 45.33%**, **Non-Causal F1 = 96.92%**.
    - **Verified Confusion Matrix (reconciles exact 1,408 test examples):**
      - $\text{TN} = 1,292$, $\text{FP} = 16$, $\text{FN} = 66$, $\text{TP} = 34$ (Sum = $1,408$).
  - **DAG Construction & Graph Audit (NetworkX):**
    - Constructs directed graphs using stable identifier pairs `(section_id, annotation_id)` as node keys, preserving display text as node attributes.
    - Dynamic edge provenance (`source_article`, `section_id`, `candidate_id`, `between_text`, `origin: Logistic Regression (Unweighted)`).
    - Accumulates evidence records on edges for parallel extractions.
    - Audits self-loops, duplicate edges, conflicting directions, and directed cycles.
    - Exact graph evaluation on test article `PMC2714965` case study:
      - Predicted Edges: 6, Gold Standard Edges: 22
      - Edge True Positives: 6, Edge False Positives: 0, Edge False Negatives: 16
      - **Edge Precision = 100.00% (6/6)**, **Edge Recall = 27.27% (6/22)**, **Edge F1 = 42.86%**.
    - Corpus-wide test graph evaluation (all 4 held-out test articles):
      - Predicted Edges: 50, Gold Standard Edges: 100
      - Edge True Positives: 34, Edge False Positives: 16, Edge False Negatives: 66
      - **Corpus Edge Precision = 68.00% (34/50)**, **Corpus Edge Recall = 34.00% (34/100)**, **Corpus Edge F1 = 45.33%**.
  - **Probabilistic Modeling & Causal Queries (pgmpy 1.1.2):**
    - Clarifies that text extraction provides qualitative graph topology, not empirical joint distributions; modeled as an illustrative synthetic Bayesian network.
    - Constructs a valid synthetic `DiscreteBayesianNetwork` with explicit TabularCPDs for the extracted `PmrB` -> `PmrA` <- `PmrD`, `PmrA` -> `Polymyxin_Resistance` pathway.
    - Evaluates baseline marginals ($P=0.3590$) and observational conditioning ($P(\text{Resistance}=1 \mid \text{PmrB}=1) = 0.6600$).
    - Simulates Pearl's interventional $do$-operator via graph surgery (`model.do(['PmrA'])`), yielding pure causal effect $P(\text{Resistance}=1 \mid do(\text{PmrA}=1)) = 0.8500$.
  - **Supplementary Corr2Cause Classifier:**
    - Evaluates classical classifier on Corr2Cause tabular features (**Accuracy = 55.34%**, **Macro-F1 = 50.51%**), clearly labeled as an isolated formal entailment benchmark.

- **`AML/AML_05_integration_evaluation.ipynb`**:
  - **End-to-End Pipeline Integration & Common Output Schema:**
    - Establishes a canonical 17-column `UnifiedRelationRecord` schema (`relation_id`, `article_id`, `section_id`, `cause_id`, `cause_text`, `cause_start`, `cause_end`, `effect_id`, `effect_text`, `effect_start`, `effect_end`, `direction`, `confidence`, `extraction_method`, `evidence_text`, `provenance`, `split`).
  - **Master Multi-Pathway Benchmark (Held-Out Test Split — 100 Gold Relations):**
    - **Pathway 1 (Rule & Dependency on Gold Argument Spans — Shared `causal_extractors.py`):**
      - Lexical Rule Baseline: Precision = **40.74%**, Recall = **33.00%**, F1 = **36.46%** (81 predictions, 33 TP, 48 FP, 67 FN).
      - spaCy Dependency Extractor: Precision = **34.72%**, Recall = **25.00%**, F1 = **29.07%** (72 predictions, 25 TP, 47 FP, 75 FN).
      - Hybrid Extractor: Precision = **37.93%**, Recall = **44.00%**, F1 = **40.74%** (116 predictions, 44 TP, 72 FP, 56 FN).
    - **Pathway 2 (Supervised Classical Classifier on Gold Argument Spans):**
      - Validation-selected `Logistic Regression (Unweighted)`: Precision = **68.00%**, Recall = **34.00%**, F1 = **45.33%** (50 predictions, 34 TP, 16 FP, 66 FN).
    - **Pathway 3 (Genuinely End-to-End Extraction from Raw Text via Predicted Mentions):**
      - Full unassisted cascade from raw text (spaCy NER $\to$ candidate pairing $\to$ LR classification).
      - Evaluated on test articles using exact span boundaries and relaxed overlap ($\text{IoU} \ge 0.5$).
      - Yields Precision = **0.00%**, Recall = **0.00%**, F1 = **0.00%** (44 candidate relations predicted, 0 true positive overlap with situation spans).
      - **Critical Methodological Finding:** Empirically demonstrates the error propagation bottleneck of off-the-shelf general NER (`en_core_web_sm`) on complex biomedical situation descriptions, providing clear justification for domain-specific fine-tuning (e.g. BioBERT / PubMedBERT in `DL/`).
  - **Directed Graph (DAG) Compilation:**
    - Single-article case study (`PMC2714965`): Edge Precision = **100.00% (6/6)**, Recall = **27.27% (6/22)**, Edge F1 = **42.86%**, Acyclic = True.
    - Corpus-wide test partition (all 4 test articles): Edge Precision = **68.00% (34/50)**, Recall = **34.00% (34/100)**, Edge F1 = **45.33%**, Acyclic = True.
  - **Probabilistic Bayesian Network Reasoning:**
    - Transparently models the neuro-symbolic bridge: text mining extracts qualitative graph topology, while numerical inference demonstrates Pearl's $do$-calculus ($P=0.8500$) vs. observational conditioning ($P=0.6500$) on explicit synthetic CPDs.

---

## Dataset Partition Summary

| Dataset Partition | PMC Articles | Section Files (Raw) | Sections with Candidates | Gold Relations | Candidate Pairs | Causal Prevalence (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Split** | 12 | 130 | 121 | 621 | 10,086 | 6.16% |
| **Validation Split** | 4 | 33 | 33 | 131 | 1,914 | 6.84% |
| **Held-Out Test Split**| 4 | 35 | 31 | 100 | 1,408 | 7.10% |
| **Total BioCause** | **20** | **198** | **185** | **852** | **13,408** | **6.35%** |

---

## Environment Setup & Dependency Verification

The project is tested and verified on **Python 3.14.2 macOS arm64 (Apple Silicon)**.

### Quickstart

```bash
# 1. Activate project virtual environment
source .venv/bin/activate

# 2. Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# 3. Install active pipeline dependencies
pip install -r requirements.txt

# 4. Download spaCy English model for NLP extraction
python -m spacy download en_core_web_sm

# 5. (Optional) Install deep learning dependencies for DL/ notebooks
pip install -r requirements-dl.txt
```

### Dependency Roles

- **`requirements.txt`**: Active dependencies for `00_setup.ipynb`, `01_eda.ipynb`, and `AML/AML_01` through `AML_05` (`pandas`, `numpy`, `datasets`, `networkx`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `ipykernel`, `spacy`, `pgmpy`, `nbclient`, `nbformat`).
- **`requirements-dl.txt`**: Optional dependencies for deep learning workflows in `DL/` (`torch`, `transformers`, `accelerate`).

---

## Generated Dataset Artifacts (Ignored by Git)

All raw corpora and generated tables are stored in git-ignored locations conforming to `.gitignore`:

| Path | Task / Role | Records | Description |
| :--- | :--- | :--- | :--- |
| `data/raw/biocause.zip` | BioCause Raw Archive | 1.3 MB | Official NaCTeM distribution archive |
| `data/raw/biocause/BioCause_corpus/` | BioCause Raw Standoff | 198 `.txt`, 198 `.ann` | Extracted full-text sections and standoff annotations |
| `data/raw/train.csv` | Corr2Cause Raw Train | 205,734 rows × 4 cols | Downloaded benchmark train split |
| `data/raw/validation.csv` | Corr2Cause Raw Val | 1,076 rows × 4 cols | Downloaded benchmark validation split |
| `data/raw/test.csv` | Corr2Cause Raw Test | 1,162 rows × 4 cols | Downloaded benchmark test split |
| `data/processed/train_val.csv` | Corr2Cause Combined | 206,810 rows × 4 cols | Combined train + validation splits |
| `data/processed/test.csv` | Corr2Cause Processed Test | 1,162 rows × 4 cols | Canonical test partition |
| `data/processed/biocause_relations.csv` | Primary: BioCause Gold Relations | 852 rows × 19 cols | Full gold relation triples with character offsets |
| `data/processed/biocause_entities.csv` | Primary: BioCause Gold Mentions | 12,939 rows × 10 cols | Gold entities, situation spans, and triggers |
| `data/processed/biocause_candidates.csv` | Primary: BioCause Candidates | 13,408 rows × 17 cols | Candidate argument pairs for relation classification |
| `data/processed/train_val_features.csv` | Supplementary: Corr2Cause Features | 206,810 rows × 26 cols | 19 engineered features for Corr2Cause |
| `data/processed/test_features.csv` | Supplementary: Corr2Cause Test Features | 1,162 rows × 25 cols | 19 engineered features for test set |

---

## Reproducible Sequential Execution Order

The AML pipeline must be executed sequentially from fresh kernels:

1. **`00_setup.ipynb`**: Acquires and validates BioCause corpus; verifies and transactionally stages Corr2Cause splits.
2. **`01_eda.ipynb`**: Conducts exploratory data analysis across partitions for both BioCause and Corr2Cause.
3. **`AML/AML_01_data_features.ipynb`**: Parses BioCause, executes deterministic group splitting, exports standardized relation and candidate tables, and extracts Corr2Cause features.
4. **`AML/AML_02_ner.ipynb`**: Validates entity offsets, tests conservative canonicalization, and benchmarks spaCy NER under Span-Only, Typed, and Maximum Relaxed Matching.
5. **`AML/AML_03_dependency_patterns.ipynb`**: Evaluates rule-based and dependency extraction patterns on held-out BioCause splits with strict directional correctness and negation controls.
6. **`AML/AML_04_classical_classifier.ipynb`**: Trains classical classifiers on candidate pairs, compiles stable-ID NetworkX DAGs, runs graph integrity audits, and executes observational and interventional queries in `pgmpy`.
7. **`AML/AML_05_integration_evaluation.ipynb`**: Integrates the end-to-end pipeline, evaluates multi-pathway extraction benchmarks (Gold-conditioned vs. Raw End-to-End), audits test DAGs, and formalizes the neuro-symbolic Bayesian network bridge.

---

## Known Limitations and Remaining DL Work

1. **Corpus Size Sensitivity:** BioCause contains 20 PMC source articles. Splitting 12/4/4 provides complete document isolation, but document-level variance across the 4 test articles affects test metrics.
2. **Supplied Argument Conditionality:** In `AML_04`, relation classification is evaluated conditional on gold argument spans (classifying candidate pairs $(Arg_1, Arg_2)$), rather than joint end-to-end extraction from unstructured raw text.
3. **Synthetic Probability Demonstration:** Textual relation extraction identifies qualitative DAG topology ($PmrB \to PmrA \to Resistance$), but does not provide empirical joint patient probability distributions; interventional $do$-calculus is demonstrated using an illustrative synthetic Bayesian network.
4. **Future Deep Learning Pipeline (`DL/`):**
   - Fine-tuning pretrained biomedical transformers (e.g., BioBERT, PubMedBERT) on `biocause_candidates.csv` for high-precision causal relation classification.
   - Comparing classical AML classifiers against neural representations on held-out test articles.
   - Feeding reviewed transformer extractions into the NetworkX and `pgmpy` DAG reasoning pipeline.
