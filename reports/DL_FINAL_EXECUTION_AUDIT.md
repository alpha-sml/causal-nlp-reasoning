# Final DL Notebook Execution and Independent Audit Report

- **Repository:** `alpha-sml/causal-nlp-reasoning`
- **Branch:** `main`
- **Date of Audit:** `2026-11-16`
- **Status:** COMPLETE & RECONCILED FROM GOOGLE COLAB EXECUTION OUTPUTS

---

## A. Execution Summary

### 1. Environment & Hardware Verification
- **Runtime Environment:** Google Colab / Python 3 (`ipykernel`) kernel.
- **Kernel Specification:**
  ```json
  "kernelspec": {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3"
  }
  ```
- **Execution Mode:** Manual top-to-bottom sequential execution of all five Deep Learning notebooks (`DL_01` through `DL_05`) in Google Colab in full dependency order, reconciled with static JSON/source inspection of all executed notebook outputs (`DL/dl_utils.py` blob SHA `1844e06924e6b971c749f4453928c66427ccc9fc`).

### 2. Notebook Execution Register
| Notebook | Scope | Total Cells | Code Cells | MD Cells | Outputs Present | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `DL/DL_01_data_preparation.ipynb` | Tokenization, entity marking, BioCause split manifest creation | 33 | 21 | 12 | 13 | **Executed** |
| `DL/DL_02_corr2cause_distilbert.ipynb` | DistilBERT fine-tuning and evaluation on Corr2Cause | 14 | 9 | 5 | 16 | **Executed** |
| `DL/DL_03_biocause_pubmedbert.ipynb` | PubMedBERT fine-tuning, joint epoch/threshold selection, BioCause test | 14 | 9 | 5 | 22 | **Executed** |
| `DL/DL_04_hybrid_graph_integration.ipynb` | Validation-only fusion grid search, frozen test evaluation, DAG extraction | 18 | 9 | 9 | 8 | **Executed** |
| `DL/DL_05_evaluation_ablation.ipynb` | Global benchmark evaluation, 4D ablations, causal Bayesian reasoning | 13 | 8 | 5 | 38 | **Executed** |
| **TOTAL** | | **92** | **56** | **36** | **97** | **All Executed** |

---

## B. Notebook Cleanup and Privacy Audit

### 1. Comprehensive Path Disclosure Audit
Across all five DL notebooks and this audit report:
- Zero local machine absolute paths (`file:///`, `/Users/...`), VS Code resource URLs, or local machine usernames are referenced in source markdown or deliverables.
- All file references in this report use repository-relative paths exclusively (e.g. `DL/DL_01_data_preparation.ipynb`, `reports/dl_final_benchmark_summary.csv`).

### 2. Preservation of Scientific Research Outputs
All retained notebook outputs reflect genuine Google Colab executions:
- Precision, recall, positive-class F1, Macro-F1, ROC-AUC, and PR-AUC across splits.
- Exact confusion matrices for all evaluated models and disaggregated variable counts.
- Complete 7-row executed ablation summary (`reports/dl_ablation_summary.csv`) including checkpoint and epoch provenance.
- NetworkX graph statistics, deterministic cycle-pruning counts, and acyclicity verification.
- Illustrative `pgmpy` Bayesian Network observational and interventional $do$-calculus query results.

---

## C. Scientific Results and Metric Verification

### 1. Data Integrity and Split Validation
- **Corr2Cause Splits:**
  - Train: `205,734`
  - Validation: `1,076`
  - Test: `1,162`
  - Maximum Sequence Length: `256` tokens
- **Authoritative BioCause Model-Eligible Counts** (excluding `202` invalid standoff/budget reconstructions from `13,408` raw candidates):
  - Train: `9,928` (`621` positive, `9,307` negative; dynamic positive class weight $w_{\text{pos}} = 9,307 / 621 \approx 14.9871$)
  - Validation: `1,884` (`131` positive, `1,753` negative)
  - Test: `1,394` (`100` positive, `1,294` negative)
- **Entity Boundary Markers & Token Budget:** Explicit tokens `[E1]`, `[/E1]`, `[E2]`, `[/E2]` injected around candidate spans; maximum sequence length capped at `384` tokens.

### 2. Corr2Cause Benchmark (DistilBERT — `DL_02` & `DL_05`)
- **Model:** `distilbert-base-uncased` fine-tuned on causal NLI premise-hypothesis pairs.
- **Validation Selection:** Epoch `3.0` (`validation_loss = 0.3651`, `validation_accuracy = 0.8745`, `validation_precision = 0.7137`, `validation_recall = 0.6277`, `validation_macro_f1 = 0.6541`, `validation_roc_auc = 0.8440`, `validation_pr_auc = 0.3895`).
- **Held-Out Test Performance ($N = 1,162$):**
  - **Accuracy:** $0.8623$ ($1002 / 1162$)
  - **Positive (Class 1 Entailed):** Precision = $0.5962$, Recall = $0.3444$, Pos F1 = $0.4366$
  - **Macro Metrics:** Precision = $0.7423$, Recall = $0.6508$, Macro-F1 = $0.6791$
  - **ROC-AUC:** $0.8594$ | **PR-AUC:** $0.5135$
  - **Confusion Matrix** (`[[TN, FP], [FN, TP]]`):
    $$\begin{pmatrix} 940 & 42 \\ 118 & 62 \end{pmatrix}$$
- **Disaggregated by `num_variables`:**
  | Scope | Examples | TP | FP | FN | TN | Accuracy | Pos Precision | Pos Recall | Pos F1 | Macro F1 | ROC-AUC | PR-AUC |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | `num_variables = 2` | 6 | 0 | 1 | 0 | 5 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.4545 | - | - |
  | `num_variables = 3` | 48 | 3 | 8 | 0 | 37 | 0.8333 | 0.2727 | 1.0000 | 0.4286 | 0.6655 | 0.9111 | 0.5278 |
  | `num_variables = 4` | 72 | 0 | 8 | 0 | 64 | 0.8889 | 0.0000 | 0.0000 | 0.0000 | 0.4706 | - | - |
  | `num_variables = 5` | 514 | 31 | 17 | 26 | 440 | 0.9163 | 0.6458 | 0.5439 | 0.5905 | 0.7719 | 0.9317 | 0.6806 |
  | `num_variables = 6` | 522 | 28 | 8 | 92 | 394 | 0.8084 | 0.7778 | 0.2333 | 0.3590 | 0.6232 | 0.7539 | 0.5295 |
- **Data Isolation:** Strict isolation maintained; Corr2Cause data is never mixed into BioCause biomedical relation extraction or graph synthesis.

### 3. BioCause Relation Extraction Benchmark (`DL_03`–`DL_05`)
Evaluated on identical model-eligible test candidates ($N = 1,394$; $100$ positive, $1,294$ negative across 4 held-out test PMC articles):

- **Lexical Baseline (`between_text` regex baseline):**
  - **Accuracy:** $0.6901$ ($962 / 1394$)
  - **Pos Precision:** $0.0202$ | **Pos Recall:** $0.0700$ | **Pos F1:** $0.0314$ | **Macro-F1:** $0.4235$
  - **Confusion Matrix** (`[[TN, FP], [FN, TP]]`): `[[955, 339], [93, 7]]`

- **AML Baseline (`aml_biocause_*_predictions.csv`, Fixed $\tau = 0.5000$):**
  - **Validation ($N = 1,884$):** `129` positive predictions, Pos Precision = $0.5969$, Pos Recall = $0.5878$, Pos F1 = $0.5923$, Macro-F1 = $0.7810$.
  - **Held-Out Test ($N = 1,394$):** Accuracy = $0.9419$ ($1313 / 1394$), Pos Precision = $0.6118$, Pos Recall = $0.5200$, Pos F1 = $0.5622$, Macro-F1 = $0.7655$, ROC-AUC = $0.9431$, PR-AUC = $0.6119$.
  - **Test Confusion Matrix** (`[[TN, FP], [FN, TP]]`): `[[1261, 33], [48, 52]]`

- **AML-Only Validation-Selected ($\tau^* = 0.5200$):**
  - **Validation ($N = 1,884$):** `120` positive predictions, Pos Precision = $0.6250$, Pos Recall = $0.5725$, Pos F1 = $0.5976$, Macro-F1 = $0.7844$.
  - **Held-Out Test ($N = 1,394$):** Accuracy = $0.9440$ ($1316 / 1394$), Pos Precision = $0.6447$, Pos Recall = $0.4900$, Pos F1 = $0.5568$, Macro-F1 = $0.7635$, ROC-AUC = $0.9431$, PR-AUC = $0.6119$.
  - **Test Confusion Matrix** (`[[TN, FP], [FN, TP]]`): `[[1267, 27], [51, 49]]`

- **PubMedBERT Classifier (Joint Validation-Selected `checkpoint-3723`, Epoch `3.0`, $\tau^* = 0.0273$):**
  - **Validation Probability Calibration Summary ($N = 1,884$):** min = $0.0013$, median = $0.0059$, mean = $0.0335$, max = $0.3182$.
  - **Validation Metrics at $\tau^* = 0.0273$:** `377` positive predictions, Pos Precision = $0.2493$, Pos Recall = $0.7176$, Pos F1 = $0.3701$, Macro-F1 = $0.6360$, ROC-AUC = $0.8588$, PR-AUC = $0.3204$, Validation Confusion Matrix (`[[TN, FP], [FN, TP]]`): `[[1470, 283], [37, 94]]`. *(At fixed $\tau = 0.5000$: `0` positive predictions, Pos F1 = $0.0000$, Macro-F1 = $0.4820$.)*
  - **Held-Out Test at Frozen (`checkpoint-3723`, $\tau^* = 0.0273$):** Accuracy = $0.8221$ ($1146 / 1394$), Pos Precision = $0.2597$, Pos Recall = $0.8000$, Pos F1 = $0.3922$, Macro-F1 = $0.6440$, ROC-AUC = $0.8809$, PR-AUC = $0.3130$.
  - **Test Confusion Matrix** (`[[TN, FP], [FN, TP]]`): `[[1066, 228], [20, 80]]`

### 4. Hybrid Integration, Graph Refinement & Illustrative Do-Calculus (`DL_04` & `DL_05`)
- **Fusion Formula:**
  $$P_{\text{hybrid}}^{(i)} = \alpha \cdot P_{\text{DL}}^{(i)} + (1 - \alpha) \cdot P_{\text{AML}}^{(i)}$$
- **Validation Parameter Selection ($N = 1,884$):**
  - Grid search over $\alpha \in [0.00, 1.00]$ (step $0.05$) and candidate thresholds $\tau \in [0.01, 0.99]$ + validation quantiles with positive-F1-first objective.
  - Optimal validation configuration: $\alpha^* = 0.45$, $\tau^* = 0.3200$.
  - Validation metrics at $(\alpha^* = 0.45, \tau^* = 0.3200)$: `119` positive predictions, Pos Precision = $0.6723$, Pos Recall = $0.6107$, Pos F1 = **$0.6400$**, Macro-F1 = **$0.8072$** ($+0.0424$ validation Pos F1 over tuned AML).
- **Held-Out Test Evaluation ($N = 1,394$, Parameters Frozen at $\alpha^* = 0.45, \tau^* = 0.3200$):**
  - **Accuracy:** $0.9433$ ($1315 / 1394$)
  - **Pos Precision:** $0.6129$ | **Pos Recall:** $0.5700$
  - **Pos F1:** **$0.5907$** ($+0.0339$ vs. tuned AML `0.5568`, $+0.0285$ vs. fixed AML `0.5622`, $+0.1985$ vs. standalone PubMedBERT `0.3922`)
  - **Macro-F1:** **$0.7801$** | **ROC-AUC:** **$0.9448$** | **PR-AUC:** **$0.6364$**
  - **Test Confusion Matrix** (`[[TN, FP], [FN, TP]]`): `[[1258, 36], [43, 57]]`
- **Post-Processing & Graph Refinement:**
  1. **Linguistic Sanity Filter:** `93` initial positive predictions $\to$ `7` rejected by negation (`0` by distance, `0` both) $\to$ `86` surviving positive relations (Test Accuracy = $0.9426$, Pos Precision = $0.6163$, Pos Recall = $0.5300$, Pos F1 = $0.5699$, Macro-F1 = $0.7696$, Confusion Matrix `[[1261, 33], [47, 53]]`).
  2. **Deterministic Cycle Breaking (`enforce_dag_acyclicity`):** `127` nodes, `86` raw edges (`Is DAG = False`) $\to$ pruned `7` bidirectional conflict edges (`0` self-loops, `0` cycle feedback edges) $\to$ `79` acyclic directed edges (`Is DAG = True`).
  3. **Graph Edge Evaluation ($100$ Gold Test Relations):**
     - **Raw Extracted Graph (Pre-Pruning):** `86` edges, `TP = 53, FP = 33, FN = 47`, Precision = $0.6163$, Recall = $0.5300$, F1 = $0.5699$, `Is DAG = False`.
     - **Acyclic Causal DAG (Post-Pruning):** `79` edges, `TP = 53, FP = 26, FN = 47`, Precision = **$0.6709$**, Recall = $0.5300$, F1 = **$0.5922$**, `Is DAG = True`. *(All 7 bidirectional conflict removals eliminated false positives while retaining all 53 true positives.)*
- **Illustrative Bayesian Network Reasoning (`pgmpy`, `DL_05` Section 7):**
  - 3-node biological motif (`Sensor -> Regulator -> Resistance`) with explicitly synthetic CPDs:
  - Baseline marginal $P(\text{Resistance}=1) = 0.4242$
  - Observational $P(\text{Resistance}=1 \mid \text{Sensor}=1) = 0.7200$ (`+0.2958`)
  - Interventional $P(\text{Resistance}=1 \mid do(\text{Regulator}=1)) = 0.8500$ (`+0.4258`)
  - *Scientific Note:* CPDs are synthetic and illustrative; they demonstrate Pearl's $do$-calculus graph surgery separately from empirical corpus co-occurrence counts.

### 5. Multi-Dimensional Ablation Verification (`reports/dl_ablation_summary.csv`)
All variants were evaluated on identical held-out test candidates ($N = 1,394$) with validation-only joint checkpoint and threshold selection:

| Ablation | Status | Selected Checkpoint | Epoch | Threshold | Val Pos F1 | Val Macro F1 | TP | FP | FN | TN | Accuracy | Pos Precision | Pos Recall | Pos F1 | Macro F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Full model with markers and weighted loss` | `completed` | `checkpoint-3723` | `3.0` | `0.0273` | 0.3701 | 0.6360 | 80 | 228 | 20 | 1066 | 0.8221 | 0.2597 | **0.8000** | 0.3922 | 0.6440 | 0.8809 | 0.3130 |
| `No entity markers` | `completed` | `checkpoint-3723` | `3.0` | `0.0566` | 0.2356 | 0.5682 | 50 | 226 | 50 | 1068 | 0.8020 | 0.1812 | 0.5000 | 0.2660 | 0.5758 | 0.7924 | 0.2108 |
| `Entity markers and unweighted loss` | `completed` | `checkpoint-2482` | `2.0` | `0.0439` | 0.4667 | 0.7103 | 60 | 104 | 40 | 1190 | 0.8967 | 0.3659 | 0.6000 | 0.4545 | 0.6987 | 0.8967 | 0.3880 |
| `Pure AML (fixed 0.50)` | `completed` | `-` | `-` | `0.5000` | - | - | 52 | 33 | 48 | 1261 | 0.9419 | 0.6118 | 0.5200 | 0.5622 | 0.7655 | 0.9431 | 0.6119 |
| `AML-only validation-selected threshold` | `completed` | `-` | `-` | `0.5200` | 0.5976 | 0.7844 | 49 | 27 | 51 | 1267 | **0.9440** | **0.6447** | 0.4900 | 0.5568 | 0.7635 | 0.9431 | 0.6119 |
| `Validation-selected hybrid` | `completed` | `-` | `-` | `0.3200` | **0.6400** | **0.8072** | 57 | 36 | 43 | 1258 | 0.9433 | 0.6129 | 0.5700 | **0.5907** | **0.7801** | **0.9448** | **0.6364** |
| `Sanitized hybrid (linguistic filtering)` | `completed` | `-` | `-` | `0.3200` | - | - | 53 | 33 | 47 | 1261 | 0.9426 | 0.6163 | 0.5300 | 0.5699 | 0.7696 | 0.9448 | 0.6364 |

Key ablation findings:
1. **Entity Markers (`[E1]`, `[E2]`):** Removing explicit argument markers degrades test Pos F1 from `0.3922` to `0.2660` (`-0.1262`), ROC-AUC from `0.8809` to `0.7924`, and PR-AUC from `0.3130` to `0.2108`.
2. **Dynamic Class Weighting vs. Calibrated Thresholding:** At fixed `0.50` (`argmax`), unweighted cross-entropy predicts `0` positives across all 3 epochs. Under joint validation epoch and threshold calibration (`checkpoint-2482`, Epoch `2.0`, $\tau = 0.0439$), unweighted CE reaches higher standalone precision (`0.3659` vs `0.2597`), Pos F1 (`0.4545` vs `0.3922`), and PR-AUC (`0.3880` vs `0.3130`), while dynamic weighting ($w_{\text{pos}} = 14.9871$) achieves much higher positive recall (`0.8000` vs `0.6000`).
3. **Hybrid Ensembling:** Validation-tuned linear fusion ($\alpha^* = 0.45, \tau^* = 0.3200$) outperforms every standalone model on Pos F1 (`0.5907`), Macro-F1 (`0.7801`), ROC-AUC (`0.9448`), and PR-AUC (`0.6364`).
4. **Linguistic Post-Processing & DAG Constraints:** Linguistic negation filtering removes 4 TPs alongside 3 FPs at candidate level (`Pos F1: 0.5907 -> 0.5699`), while subsequent `enforce_dag_acyclicity` pruning eliminates 7 strictly false-positive bidirectional conflicts, boosting graph edge Precision to `0.6709` and Graph F1 to `0.5922`.

---

## D. Static Validation & Integrity Checks

| Validation Check | Scope / Assertion | Result |
| :--- | :--- | :---: |
| Notebook JSON Validation | `jq empty` across all 5 DL notebooks (`DL_01`–`DL_05`) | **Valid JSON (Exit 0)** |
| Git Whitespace Check | `git diff --check` across working tree | **Clean (Exit 0)** |
| Shared Utilities Preservation | `git hash-object DL/dl_utils.py` | **`1844e06924e6b971c749f4453928c66427ccc9fc` (Exact Match)** |
| Candidate ID Alignment | Exact order & label agreement across BioCause val ($N=1,884$) and test ($N=1,394$) in executed Colab assertions | **100% Match (Passed in `DL_04` / `DL_05`)** |
| Probability Bounds | Finite probabilities in $[0, 1]$ across AML, PubMedBERT, and Hybrid streams | **Verified (Passed in `DL_03`–`DL_05`)** |
| DAG Acyclicity | `nx.is_directed_acyclic_graph(G_dag)` & edge-count pruning reconciliation (`86 - 7 = 79`) | **`True` (Passed in `DL_04` & `DL_05`)** |
| Phase 1 Protection | Unmodified `AML/`, `00_setup.ipynb`, `01_eda.ipynb`, `tests/test_phase1_baseline.py`, `requirements.txt` | **Unmodified** |

---

## E. Outstanding Issues and Limitations

1. **Synthetic CPDs:** The Bayesian Network demonstration in Section 7 of `DL_05` (`Sensor -> Regulator -> Resistance`) and Section 8 of `DL_04` uses synthetic conditional probability distributions (CPDs) to illustrate Pearl's $do$-calculus and must not be interpreted as empirically estimated biological effect sizes.
2. **Low Probability Calibration in Weighted PubMedBERT:** Under dynamic positive class weighting ($14.9871\times$), raw softmax probabilities on validation remain compressed (`max = 0.3182`), requiring validation-selected thresholding ($\tau^* = 0.0273$) rather than a naive $0.50$ cutoff.
3. **Working Tree Status:** All notebook outputs, extracted report artifacts, and reconciled documentation remain unstaged in the local working tree without committing or pushing to Git.

---

## F. Final Deliverables Register

1. **Executed Notebooks:**
   - `DL/DL_01_data_preparation.ipynb`
   - `DL/DL_02_corr2cause_distilbert.ipynb`
   - `DL/DL_03_biocause_pubmedbert.ipynb`
   - `DL/DL_04_hybrid_graph_integration.ipynb`
   - `DL/DL_05_evaluation_ablation.ipynb`
2. **Reconciled Audit & Report Deliverables (`reports/`):**
   - `reports/DL_FINAL_EXECUTION_AUDIT.md`
   - `reports/dl_final_benchmark_summary.csv`
   - `reports/dl_ablation_summary.csv`
   - `reports/dl_biocause_pr_roc_curves.png`
   - `reports/dl_biocause_bayesian_dag_motif.png`
3. **Pipeline Prediction, Config & Graph Artifacts (Colab-generated in `data/processed/`):**
   - `data/processed/dl_corr2cause_test_predictions.csv`
   - `data/processed/dl_biocause_metrics.json`
   - `data/processed/dl_biocause_validation_predictions.csv`
   - `data/processed/dl_biocause_test_predictions.csv`
   - `data/processed/dl_biocause_hybrid_validation_predictions.csv`
   - `data/processed/dl_biocause_hybrid_test_predictions.csv`
   - `data/processed/dl_biocause_fusion_config.json`
   - `data/processed/dl_biocause_hybrid_positive_relations.csv`
   - `data/processed/dl_biocause_hybrid_dag_edges.csv`
   - `data/processed/dl_biocause_graph_pruning_diagnostics.json`
