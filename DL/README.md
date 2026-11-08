# Phase 2: Deep Learning & Hybrid Neuro-Symbolic Integration

This directory contains the Phase 2 (Deep Learning) architecture and pipeline structure for the **Causal NLP Reasoning** project.

Phase 2 advances beyond Phase 1's classical machine learning extractors by fine-tuning compact transformer architectures (**DistilBERT** and **PubMedBERT Fulltext**), ensembling deep contextual representations with syntactic/lexical features via validation-tuned probability fusion, and maintaining strict schema compatibility with downstream symbolic causal graphs (**NetworkX DAGs**) and interventional Bayesian networks (**`pgmpy`**).

---

## 1. Phase 2 Objectives

1. **Abstract Causal Entailment**: Fine-tune a compact transformer (**DistilBERT**) on **Corr2Cause** to evaluate deductive causal graph reasoning from natural language statistical independence statements.
2. **Biomedical Causal Relation Classification**: Fine-tune a domain-pretrained transformer (**PubMedBERT Fulltext**) on **BioCause** using explicit argument boundary markers (`[E1]...[/E1]` and `[E2]...[/E2]`) and dynamic loss weighting to classify directed cause-effect links under severe class imbalance.
3. **Validation-Tuned Hybrid Ensembling**: Integrate Phase 1 classical predictions (Logistic Regression, HistGBT, Shortest Dependency Path) with Phase 2 deep learning predictions via linear probability fusion ($P_{\text{hybrid}} = \alpha \cdot P_{\text{DL}} + (1 - \alpha) \cdot P_{\text{AML}}$), where the fusion parameter $\alpha \in [0, 1]$ and classification threshold $\tau$ are calibrated exclusively on held-out validation articles.
4. **Symbolic Integration & Deterministic Cycle Breaking**: Preserve 100% compliance with the standardized 17-column `UnifiedRelationRecord` schema, reusing Phase 1 NetworkX graph utilities for cycle pruning (`enforce_dag_acyclicity`) and compiling acyclic DAGs into `pgmpy` Bayesian networks for Pearl's $do$-calculus interventional queries.
5. **Rigorous Comparative Benchmarking**: Evaluate on identical held-out test splits to quantify whether deep learning and hybrid architectures surpass Phase 1 classical baselines.

---

## 2. Overall Architecture

```mermaid
flowchart TD
    subgraph S1["1. Data Ingestion & Preprocessing (DL_01)"]
        D1["Corr2Cause Canonical CSVs<br/>Train: 205,734 | Val: 1,076 | Test: 1,162"] --> P1["Premise-Hypothesis Tokenizer Formatter"]
        D2["BioCause Candidates<br/>12 Train / 4 Val / 4 Test Articles"] --> P2["Argument Boundary Marker Injection<br/>[E1] Arg1 [/E1] ... [E2] Arg2 [/E2]<br/>(Dynamic Pos Weight ≈ 15.24)"]
    end

    subgraph S2["2. Transformer Fine-Tuning"]
        P1 --> M1["DL_02: DistilBERT Sequence Classifier<br/>(Corr2Cause NLI Entailment)"]
        P2 --> M2["DL_03: PubMedBERT Fulltext Classifier<br/>(Weighted Cross-Entropy Loss)"]
    end

    subgraph S3["3. Hybrid Fusion & Graph Integration (DL_04)"]
        M2 -->|"DL Probabilities (P_DL)"| H1["Validation-Selected Probability Fusion<br/>P_hybrid = α · P_DL + (1 - α) · P_AML<br/>(α* tuned on 4 Val Articles)"]
        AML["Phase 1 AML Stream<br/>(aml_biocause_*.csv)"] -->|"AML Probabilities (P_AML)"| H1
        H1 --> L1["Linguistic Sanity Filter<br/>(Negation & Distance Pruning)"]
        L1 --> R1["17-Column UnifiedRelationRecord Schema"]
        R1 --> G1["NetworkX Directed Graph<br/>enforce_dag_acyclicity()"]
        G1 --> G2["pgmpy DiscreteBayesianNetwork<br/>P(Effect | do(Cause)) vs P(Effect | Cause)"]
    end

    subgraph S4["4. Benchmark Evaluation & Ablation (DL_05)"]
        M1 -->|"Corr2Cause Test Preds"| E1["Corr2Cause Benchmark Section<br/>(Acc, Macro-F1 by Variable Count)"]
        H1 & M2 & AML -->|"BioCause Test Preds"| E2["BioCause Benchmark Section<br/>(P, R, F1, PR-AUC, ROC-AUC)"]
        G1 & G2 -->|"DAG & Intervention Artifacts"| E3["Graph Edge Evaluation & Interventional Analysis"]
    end
```

---

## 3. Strict Boundary: Corr2Cause vs. BioCause

A fundamental architectural constraint of this repository is the complete operational and conceptual separation between the two datasets:

| Dimension | Corr2Cause | BioCause |
| :--- | :--- | :--- |
| **Domain** | Abstract propositional causal inference | Applied biomedical literature (PubMed Central) |
| **Task Formulation** | Natural Language Inference (Premise $\to$ Hypothesis) | Candidate-conditioned directed relation classification |
| **Input Representation** | `Premise: ... Hypothesis: ...` | Sentence context with markers: `[E1] Arg1 [/E1] ... [E2] Arg2 [/E2]` |
| **Target Labels** | Binary: `1` (Entailed), `0` (Refuted) | Binary: `1` (Arg1 causes Arg2), `0` (Arg1 does not cause Arg2) |
| **Canonical Split Distribution** | Train: 205,734 \| Val: 1,076 \| Test: 1,162 | **Article-level grouping**: 12 train (10,086 cands) \| 4 val (1,914 cands) \| 4 test (1,408 cands) |
| **Downstream Graph** | **Does NOT feed into knowledge graph** | **Feeds directly into NetworkX DAG and pgmpy BN** |
| **Evaluation Scope** | Standalone benchmark table (Macro-F1, Accuracy) | Standalone relation & graph edge benchmark table (P, R, F1) |

> **Critical Boundary Rule**: Corr2Cause predictions must never be mixed with BioCause predictions, and Corr2Cause outputs must never feed into the BioCause knowledge graph.

---

## 4. Technical Scope of Notebooks

| Notebook | Scope & Core Technical Responsibilities |
| :--- | :--- |
| **`DL_01_data_preparation.ipynb`** | Ingest canonical raw datasets; audit 12/4/4 article-level split isolation (zero cross-split leakage); inject argument boundary markers (`[E1]`, `[/E1]`, `[E2]`, `[/E2]`); compute training dynamic class weight ($w_{\text{pos}} = 9,465 / 621 \approx 15.24$); profile token lengths; serialize model-ready splits. |
| **`DL_02_corr2cause_distilbert.ipynb`** | Fine-tune `distilbert-base-uncased` for Corr2Cause binary causal entailment; format `[CLS] Premise [SEP] Hypothesis [SEP]`; AdamW optimization with FP16; checkpoint selection on validation split; export test predictions. |
| **`DL_03_biocause_pubmedbert.ipynb`** | Fine-tune `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` for BioCause directed relation classification; resize embeddings for special boundary tokens; apply dynamic class-weighted loss; tune classification threshold on validation articles; export test probabilities. |
| **`DL_04_hybrid_graph_integration.ipynb`** | Ingest AML and PubMedBERT streams; grid-search validation-optimal fusion weight $\alpha^* \in [0, 1]$ and threshold $\tau^*$; apply linguistic negation filtering; format positive predictions into 17-column `UnifiedRelationRecord` schema; compile NetworkX DAG with `enforce_dag_acyclicity`; execute `pgmpy` Bayesian network interventional queries. |
| **`DL_05_evaluation_ablation.ipynb`** | Comparative benchmarking (AML vs. DL vs. Hybrid); evaluate separate Corr2Cause and BioCause sections; execute 4-dimensional ablation studies; evaluate graph edge precision/recall/F1 against gold BioCause relations; generate final summary tables and figures in `reports/`. |

---

## 5. Pipeline Execution Order & Data Contracts

Notebooks are designed to execute sequentially from `DL_01` to `DL_05`:

```text
DL_01_data_preparation.ipynb
        │
        ├──► DL_02_corr2cause_distilbert.ipynb ──┐
        │                                        │
        └──► DL_03_biocause_pubmedbert.ipynb     │
                     │                           │
                     ▼                           │
             DL_04_hybrid_graph_integration.ipynb │
                     │                           │
                     ▼                           ▼
             DL_05_evaluation_ablation.ipynb ◄───┘
```

### Input and Output Artifact Contracts

```
[DL_01: Data Preparation]
  Inputs:
    - data/raw/train.csv, validation.csv, test.csv (Canonical Corr2Cause: 205,734 / 1,076 / 1,162)
    - data/processed/biocause_candidates.csv (BioCause candidates: 10,086 train / 1,914 val / 1,408 test)
    - data/processed/biocause_relations.csv (Gold relations: 852 total, 100 test)
  Outputs:
    - data/processed/dl_corr2cause_{train,val,test}.csv
    - data/processed/dl_biocause_{train,val,test}.csv (with [E1]...[E2] markers)

[DL_02: Corr2Cause DistilBERT]
  Inputs:
    - data/processed/dl_corr2cause_{train,val,test}.csv
  Outputs:
    - DL/checkpoints/corr2cause_distilbert/ (best model weights)
    - data/processed/dl_corr2cause_test_predictions.csv (columns: index, true_label, pred_label, prob_entailed, num_variables)

[DL_03: BioCause PubMedBERT]
  Inputs:
    - data/processed/dl_biocause_{train,val,test}.csv
  Outputs:
    - DL/checkpoints/biocause_pubmedbert/ (best model weights)
    - data/processed/dl_biocause_validation_predictions.csv (1,914 rows; columns: candidate_id, article_id, section_id, arg1_id, arg2_id, true_label, dl_prob, dl_pred, split)
    - data/processed/dl_biocause_test_predictions.csv (1,408 rows; columns: candidate_id, article_id, section_id, arg1_id, arg2_id, true_label, dl_prob, dl_pred, split)

[DL_04: Hybrid Integration & Graph Compilation]
  Inputs:
    - data/processed/aml_biocause_validation_predictions.csv (Phase 1 AML prerequisite, 1,914 rows)
    - data/processed/aml_biocause_test_predictions.csv (Phase 1 AML prerequisite, 1,408 rows)
    - data/processed/dl_biocause_validation_predictions.csv (from DL_03, 1,914 rows)
    - data/processed/dl_biocause_test_predictions.csv (from DL_03, 1,408 rows)
  Outputs:
    - data/processed/dl_biocause_hybrid_predictions.csv (17-column UnifiedRelationRecord schema)
    - Verified acyclic NetworkX DAG and pgmpy Bayesian Network

[DL_05: Comprehensive Evaluation & Ablations]
  Inputs:
    - data/processed/dl_corr2cause_test_predictions.csv (from DL_02)
    - data/processed/dl_biocause_test_predictions.csv (from DL_03)
    - data/processed/dl_biocause_hybrid_predictions.csv (from DL_04)
    - data/processed/biocause_relations.csv (Gold 100 test relations)
  Outputs:
    - reports/dl_final_benchmark_summary.csv
    - reports/dl_ablation_summary.csv
    - High-resolution evaluation figures (PR curves, ROC curves, calibration plots)
```

---

## 6. AML Prediction Export Prerequisite

`DL_04_hybrid_graph_integration.ipynb` requires candidate-aligned AML prediction files:
- `data/processed/aml_biocause_validation_predictions.csv` (1,914 rows)
- `data/processed/aml_biocause_test_predictions.csv` (1,408 rows)

Required columns:
```text
candidate_id, true_label, aml_prob, aml_pred, split
```

### Primary AML Baseline Model
The primary classical classifier designated as the input stream for hybrid integration is **`Logistic Regression (Moderate Weight 1:2)`**, which achieved the highest Phase 1 validation F1 score (55.32% F1 on test). Alternative AML models (such as `Logistic Regression (Unweighted)`, `HistGBT`, and `Random Forest`) are preserved and reported as separate comparative baselines.

### Reproducible Generation from Phase 1
The AML classifier is fitted strictly on the training partition ($N = 10,086$) and evaluated to generate held-out validation predictions ($N = 1,914$) and test predictions ($N = 1,408$). If these files are not present in `data/processed/`, they are produced by executing the unchanged Phase 1 classifier pipeline from `AML/AML_04_classical_classifier.ipynb`:
1. Load `data/processed/biocause_candidates.csv`.
2. Extract the identical 10 structural/lexical features and 100 TF-IDF n-grams fitted strictly on the training split.
3. Fit the primary classifier (`Logistic Regression (Moderate Weight 1:2)`).
4. Predict probabilities using `predict_proba()` on held-out validation candidates (1,914 rows) and test candidates (1,408 rows).
5. Export with candidate keys (`candidate_id`) to ensure 100% alignment with deep learning predictions.

---

## 7. BioCause Label Semantics & Class Imbalance

### Directional Candidate Semantics
The BioCause candidate generation pipeline extracts every co-occurring entity pair $(Arg_1, Arg_2)$ within sentences:
- **Label `1`**: Directed causal relation $Arg_1 \to Arg_2$ is annotated in the gold corpus.
- **Label `0`**: $Arg_1$ does not represent the gold directed relation $Arg_1 \to Arg_2$. **Important**: Label `0` candidates include reversed directional pairs $(Arg_2, Arg_1)$ where causality runs in the opposite direction, as well as pairs with no causal connection. Label `0` must not be mischaracterized as universally non-causal.

### Dynamic Class Imbalance Mitigation
The candidate generation process results in severe class imbalance:
- **Raw candidate audit counts**:
  - **Train split**: 621 positive / 9,465 negative = **15.24:1 negative-to-positive ratio**.
  - **Validation split**: 131 positive / 1,783 negative = **13.61:1 ratio**.
  - **Test split**: 100 positive / 1,308 negative = **13.08:1 ratio**.
- **Model-eligible source-verified counts** (excluding 222 invalid reconstructions):
  - **Train split**: 620 positive / 9,288 negative = **14.98:1 ratio**.
  - **Validation split**: 131 positive / 1,753 negative = **13.38:1 ratio**.
  - **Test split**: 100 positive / 1,294 negative = **12.94:1 ratio**.

To prevent the model from collapsing to the majority negative class, positive class loss weights must be computed dynamically strictly from model-eligible training data:
$$\text{pos\_weight} = \frac{N_{\text{eligible, train, neg}}}{N_{\text{eligible, train, pos}}} = \frac{9288}{620} \approx 14.9806$$
Weights must **never** be derived using validation or test split labels.

---

## 8. Hybrid Validation Discipline & Graph Integration

### Validation Discipline
Linear probability-score fusion combines model estimates:
$$P_{\text{hybrid}} = \alpha \cdot P_{\text{DL}} + (1 - \alpha) \cdot P_{\text{AML}}$$
- The blending parameter $\alpha \in [0, 1]$ and decision threshold $\tau$ are selected by sweeping $\alpha$ with step 0.05 strictly on the held-out validation articles (1,914 candidate pairs).
- Test set predictions are evaluated **exactly once** using the frozen validation-selected $\alpha^*$ and $\tau^*$.
- Probability-score fusion is empirically compared against both pure AML ($\alpha = 0$) and pure DL ($\alpha = 1$) baselines.

> **Note**: Raw model probability outputs are not calibrated probabilities unless an explicit calibration method (e.g., Platt scaling, isotonic regression) is applied. The fusion formula operates on raw predicted scores.

### Graph & Bayesian Network Reasoning
1. **17-Column Schema**: Positive relations are formatted into `UNIFIED_RELATION_SCHEMA` (`relation_id`, `article_id`, `section_id`, `cause_id`, `cause_text`, `cause_start`, `cause_end`, `effect_id`, `effect_text`, `effect_start`, `effect_end`, `direction`, `confidence`, `extraction_method`, `evidence_text`, `provenance`, `split`).
2. **Deterministic Cycle Breaking**: Extracted edges are processed via `enforce_dag_acyclicity()` from `AML/causal_extractors.py`, which logs and prunes self-loops, resolves bidirectional conflicts, and breaks feedback cycles.
3. **Causal Interpretation Guardrail**: An acyclic graph is a mathematical topological property, not an absolute guarantee of causal ground truth. Cycle breaking enforces formal mathematical consistency required for Bayesian inference.
4. **Bayesian Network Intervention**: Compiles acyclic DAGs into `pgmpy` `DiscreteBayesianNetwork` to illustrate observational distributions $P(Y \mid X)$ versus interventional distributions $P(Y \mid do(X))$.

---

## 9. Systematic Ablation Study Plan

The evaluation pipeline in `DL_05` executes four core ablation studies:

| Ablation Study | Experimental Configurations | Evaluation Metric | Research Question |
| :--- | :--- | :--- | :--- |
| **1. Argument Boundary Markers** | Marked (`[E1]...[E2]`) vs. Unmarked text | Precision, Recall, F1 | Do explicit positional tokens improve bidirectional attention across entity arguments? |
| **2. Dynamic Loss Weighting** | Weighted ($w_{\text{pos}} \approx 15.24$) vs. Unweighted ($w=1.0$) | Causal Recall, PR-AUC | Does dynamic weighting prevent minority class suppression under 15.24:1 imbalance? |
| **3. Ensembling Paradigm** | Pure AML ($\alpha=0$) vs. Pure DL ($\alpha=1$) vs. Hybrid ($\alpha^*$) | F1, Macro-F1, ROC-AUC | Does combining syntactic/lexical rules with contextual transformers outperform single models? |
| **4. Linguistic Post-Processing** | Hybrid with vs. without negation/distance filters | Edge Precision, False Positives | Does symbolic post-filtering effectively remove false positive causal edges? |

---

## 10. Free Kaggle GPU & Google Colab Execution Workflow

Both transformer models (`distilbert-base-uncased` and `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) are compact architectures designed to train comfortably within **free Kaggle (NVIDIA T4 x 2)** or **Google Colab (T4)** environments:

### Kaggle Environment Setup
1. **Accelerator**: Select **GPU T4 x 2** or **GPU T4**.
2. **Internet**: Toggle **Internet ON** in notebook settings to download pretrained transformer checkpoints.
3. **Pre-installed PyTorch**: Kaggle provides pre-installed PyTorch with CUDA acceleration. **Do not run `pip install torch`** as it can downgrade or conflict with the CUDA runtime.
4. **Install Phase 2 Requirements**:
   ```bash
   pip install -r requirements-dl.txt
   ```
5. **Memory and Quota Management (Conservative Initial Defaults)**:
   - BioCause batch size: 8 (with gradient accumulation: 2).
   - DistilBERT batch size: 16.
   - Mixed precision: enable `fp16=True` only when CUDA is available (do not enable on CPU or MPS).
   - Evaluation strategy: use modern `eval_strategy="epoch"` (replacing legacy `evaluation_strategy`).
   - Set `save_total_limit=1` to prevent exhausting Kaggle's 20 GB disk quota with stale checkpoints.

### Google Colab Environment Setup
1. **Runtime Type**: Navigate to `Runtime` > `Change runtime type` > select **T4 GPU**.
2. **Clone & Install**:
   ```bash
   git clone https://github.com/alpha-sml/causal-nlp-reasoning.git
   cd causal-nlp-reasoning
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   pip install -r requirements-dl.txt
   ```

### Mixed Precision & Device Selection Policy
`DL/dl_utils.py` enforces strict compute device detection and mixed-precision policies:
- Device selection follows deterministic fallback order:
  1. **NVIDIA CUDA** (`'cuda'`)
  2. **Apple MPS** (`'mps'`)
  3. **CPU** (`'cpu'`)
- **CUDA-only FP16**: `get_training_arguments()` guarantees that `fp16` is enabled **strictly on CUDA devices**.
  - CPU and Apple MPS run with FP16 **disabled** (`fp16=False`), even if an explicit `fp16=True` argument is passed, preventing runtime crashes on unsupported hardware.
  - CUDA environments automatically enable FP16 when requested or by default.
  - BF16 and other unsupported mixed-precision modes remain disabled without explicit validated implementations.

---

## 11. Dependencies & Clean Architecture Policy

Phase 2 builds upon the foundational Phase 1 AML pipeline. Because the downstream hybrid integration (`DL_04`) and evaluation modules reuse Phase 1 graph cycle pruning (`enforce_dag_acyclicity`), shortest dependency path extractors, and interventional Bayesian networks (`pgmpy`), both Phase 1 and Phase 2 dependencies are required:

### 1. Phase 1 & Shared Graph Dependencies (`requirements.txt`)
```text
pandas>=2.2.0
numpy>=1.26.0
datasets>=2.19.0
networkx>=3.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.12.0
scikit-learn>=1.4.0
ipykernel>=6.29.0
spacy>=3.7.0
pgmpy>=0.1.25
nbclient>=0.10.0
nbformat>=5.9.0
```
*(Also requires `python -m spacy download en_core_web_sm`)*

### 2. Phase 2 Deep Learning Dependencies (`requirements-dl.txt`)
```text
torch>=2.2.0
transformers>=4.41.2,<5.0.0
datasets>=2.19.0
accelerate>=0.30.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.0
pyarrow>=14.0.0
```

> **Transformers Compatibility**: `transformers>=4.41.2,<5.0.0` is specified because HuggingFace `TrainingArguments` modern API introduced `eval_strategy` (replacing the deprecated `evaluation_strategy` which raises errors in versions prior to 4.41.0).

> **Clean Architecture Policy**: No paid external APIs, proprietary LLM endpoints, heavyweight vector databases (FAISS/Chroma), RAG frameworks (LangChain/LlamaIndex), or inference servers (vLLM) are used. The pipeline relies entirely on open-source, reproducible, local transformer fine-tuning.

---

## 12. Data Directory Expectations & Fail-Fast Policy

### Required Raw Dataset Files

| File | Location | Source |
| :--- | :--- | :--- |
| `train.csv` | `data/raw/` | Corr2Cause (HuggingFace: `causalnlp/corr2cause`) |
| `validation.csv` | `data/raw/` | Corr2Cause |
| `test.csv` | `data/raw/` | Corr2Cause |
| `biocause_candidates.csv` | `data/processed/` | Generated by `00_setup.ipynb` + `AML/AML_01_data_features.ipynb` |
| `*.txt` (raw section texts) | `data/raw/biocause/BioCause_corpus/` *(or fallback `data/raw/biocause/`)* | BioCause raw standoff text files |

### Fail-Fast Missing Data Behavior
The data preparation pipeline (`DL_01_data_preparation.ipynb`) enforces a strict **fail-fast policy**:
- If canonical Corr2Cause files (`train.csv`, `validation.csv`, `test.csv`) or BioCause files (`biocause_candidates.csv` and source `.txt` files) are absent from disk, the pipeline immediately raises `FileNotFoundError` with descriptive instructions.
- The canonical pipeline **never** silently generates or substitutes synthetic/mock data.
- Synthetic fixtures are strictly confined to the offline unit test suite (`tests/test_dl_utils.py`).

### BioCause Source Context Reconstruction & Model-Eligible Filtering
In Phase 1, `biocause_candidates.csv` exports argument spans and text, but does not export full section text (it only contains `between_text`).
In Task 1:
1. `resolve_biocause_section_path` and `load_biocause_section_text` locate and load the complete source `.txt` file using `section_id` from `data/raw/biocause/BioCause_corpus/` (or fallback `data/raw/biocause/`).
2. `reconstruct_biocause_context_with_markers` verifies that character offsets match `arg1_text` and `arg2_text` (allowing documented newline normalization `c.replace("\n", " ") == arg_text`).
3. Explicit argument boundary markers `[E1]...[/E1]` and `[E2]...[/E2]` are inserted into the complete source context (never substituting `between_text`).
4. `crop_context_around_entities` crops context around both entity spans to fit within `MAX_LENGTH = 384` tokens (including special tokens). If arguments are distant, an explicit gap separator (`" ... "`) is inserted. Both complete entity spans and all four markers are guaranteed intact. If spans cannot fit or exceed the budget, a descriptive `ValueError` is raised recording the candidate ID.
5. **No Synthetic Fallback Contexts**: For candidates where source reconstruction fails, synthetic sentences or concatenated argument text from `between_text` are strictly prohibited. The audit sets `reconstruction_valid = False` and `marked_text = None`.
6. **Model-Eligible Candidate Filtering**: Invalid reconstructions are excluded from model-ready training splits because corrupt offsets or truncated argument spans compromise transformer bidirectional attention. The model-ready splits contain strictly source-verified candidates:
   - **Train**: 9,908 candidates (620 positive, 9,288 negative)
   - **Validation**: 1,884 candidates (131 positive, 1,753 negative)
   - **Test**: 1,394 candidates (100 positive, 1,294 negative)
7. **Diagnostics CSV Export**: All 222 invalid candidate pairs are exported to `data/processed/dl_biocause_reconstruction_diagnostics.csv` with their original candidate identifiers, article IDs, section IDs, offsets, and actionable error diagnostics (188 overlapping spans, 22 crop-budget limits, 12 out-of-bounds standoff offsets).
8. **Dynamic Training Loss Weight**: Recomputed strictly from model-eligible training candidates:
   $$w_{\text{pos}} = \frac{N_{\text{eligible, train, neg}}}{N_{\text{eligible, train, pos}}} = \frac{9288}{620} \approx 14.9806$$

### Expected Processed Outputs (from DL_01)

| File | Rows | Description |
| :--- | ---: | :--- |
| `dl_corr2cause_train.csv` | 205,734 | Corr2Cause training split |
| `dl_corr2cause_val.csv` | 1,076 | Corr2Cause validation split |
| `dl_corr2cause_test.csv` | 1,162 | Corr2Cause test split |
| `dl_biocause_train.csv` | 9,908 | Model-eligible BioCause training candidates with verified entity markers |
| `dl_biocause_val.csv` | 1,884 | Model-eligible BioCause validation candidates with verified entity markers |
| `dl_biocause_test.csv` | 1,394 | Model-eligible BioCause test candidates with verified entity markers |
| `dl_biocause_reconstruction_diagnostics.csv` | 222 | Complete audit diagnostics for invalid BioCause candidate pairs |

---

## 13. How to Execute the Preparation Pipeline

```bash
# 1. Install all dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -r requirements-dl.txt

# 2. Stage raw datasets
#    Run 00_setup.ipynb if data/raw/ and data/processed/ are empty

# 3. Execute data preparation notebook
jupyter nbconvert --to notebook --execute DL/DL_01_data_preparation.ipynb

# 4. Run automated test suites
pytest tests/test_dl_utils.py -v
pytest tests/test_phase1_baseline.py -v
```

---

## 14. Shared Utility Descriptions (`DL/dl_utils.py`)

| Function | Description |
| :--- | :--- |
| `find_repo_root()` | Locate repository root across local, Colab, and Kaggle environments |
| `setup_repo_imports()` | Configure `sys.path` with repo root and subdirectories |
| `set_seed(seed)` | Set random seeds for Python, NumPy, and PyTorch (CPU/GPU/MPS) |
| `get_device()` | Detect CUDA, Apple MPS, or CPU with safe fallback |
| `validate_dataset_schema(df, cols, ...)` | Validate column presence and fail fast on null/empty critical values |
| `validate_biocause_candidates(df)` | Extended BioCause validation: missing cols before access, binary labels, duplicates, split isolation |
| `format_biocause_input_with_markers(...)` | Insert `[E1]`/`[/E1]`/`[E2]`/`[/E2]` markers with offset validation |
| `crop_context_around_entities(...)` | Crop context to fit 384 tokens preserving all markers, with gap separator for distant arguments |
| `resolve_biocause_section_path(sec_id)` | Locate `.txt` section file in `BioCause_corpus` or fallback `data/raw/biocause/` |
| `load_biocause_section_text(sec_id)` | Load complete source text for a BioCause section |
| `reconstruct_biocause_context_with_markers(...)` | Load source section, verify spans with newline normalization, insert markers, crop to 384 tokens |
| `profile_token_lengths(texts, tokenizer, ...)` | Statistical token length profiling (min, max, mean, median, p95, p99, pct within budget) |
| `prepare_tokenizer_and_inputs(...)` | Load AutoTokenizer and register special entity markers |
| `calculate_dynamic_class_weight(neg, pos)` | Compute $w_{\text{pos}} = N_{\text{neg}} / N_{\text{pos}}$ from training split counts only |
| `get_training_arguments(...)` | HuggingFace TrainingArguments configured with `eval_strategy` and auto fp16 |
| `compute_classification_metrics(eval_pred)` | Accuracy, Precision, Recall, F1, Macro-F1, and Confusion Matrix |
| `fuse_probabilities(aml, dl, alpha)` | Linear probability interpolation $P_{\text{hybrid}} = \alpha \cdot P_{\text{DL}} + (1-\alpha) \cdot P_{\text{AML}}$ |
| `generate_relation_id(candidate_id)` | Deterministic SHA-256 relation identifier (`dl_{sha256[:16]}`) |
| `build_unified_relation_record(...)` | Construct 17-column UnifiedRelationRecord schema with deterministic SHA-256 relation ID |
| `export_predictions(df, path, fmt)` | Safe CSV or Parquet export with automatic parent directory creation |

---

## 15. Tokenizer Profiling & Recommended Settings

Tokenizer sequence lengths were profiled across both domains:

1. **Corr2Cause (DistilBERT)**:
   - Input format: `[CLS] Premise: {premise} Hypothesis: {hypothesis} [SEP]`
   - Mean length: ~110 tokens, p95: ~180 tokens, p99: ~220 tokens, max < 256 tokens.
   - **Recommended setting**: `MAX_LENGTH = 256` (covers >99.5% of examples without truncation).
2. **BioCause (PubMedBERT Fulltext)**:
   - Input format: `[CLS] ... [E1] Arg1 [/E1] ... [E2] Arg2 [/E2] ... [SEP]`
   - Full section texts frequently exceed 1,000 words. Following `crop_context_around_entities` with distant argument gap separation (`" ... "`), mean length is ~210 tokens, p95: ~340 tokens, p99: ~370 tokens, max $\le 384$ tokens.
   - **Recommended setting**: `MAX_LENGTH = 384` (strictly matches `DL_03_biocause_pubmedbert.ipynb`).

---

## 16. Known Limitations and Assumptions

1. **Raw datasets must be pre-staged**: The preparation notebook requires canonical data files in `data/raw/` and `data/processed/`. Run `00_setup.ipynb` and `AML_01` first.
2. **Fail-fast missing data**: If raw data is absent, the notebook fails fast with `FileNotFoundError`; no synthetic data is substituted.
3. **Tokenizer download requires internet**: Initial run requires internet connectivity to cache HuggingFace tokenizer vocabularies.
4. **Classification metrics scope**: The shared `compute_classification_metrics` helper returns point metrics at standard decision thresholds. Continuous probability curves (ROC-AUC and PR-AUC) are evaluated during comparative benchmarking in `DL_05_evaluation_ablation.ipynb`.
5. **Article-level split is inherited**: The 12/4/4 article split is inherited from Phase 1 (`00_setup.ipynb` + `AML_01`). DL_01 validates but does not reassign splits.
