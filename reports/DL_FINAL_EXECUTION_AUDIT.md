# Final DL Notebook Execution and Independent Audit Report

- **Repository:** `alpha-sml/causal-nlp-reasoning`
- **Branch:** `main`
- **Latest Reference Commit:** `a649396` (`a649396c71989fe615f2f4847bdcc40a74bfeff7`)
- **Date of Audit:** `2026-11-16`
- **Status:** EVALUATION COMPLETE; MODEL WEIGHT PERSISTENCE & `.pkl` PACKAGING PENDING COLAB VERIFICATION

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
- **Execution Mode:** Manual sequential execution of `DL_01` through `DL_05` in Google Colab, combined with static JSON, syntax, and SHA inspection in the local repository (`DL/dl_utils.py` commit `4bf3cddfedadc5c7342c6f5f1f0b04ca7bee7e5a`, blob SHA `1844e06924e6b971c749f4453928c66427ccc9fc`).
- **Completed vs. Pending Verification Boundary:**
  - **Completed in Colab:** Training, joint validation `(epoch, threshold)` selection, held-out test evaluation, prediction/metric exports (`dl_corr2cause_test_predictions.csv`, `dl_biocause_*.csv`, `dl_biocause_metrics.json`, `dl_biocause_fusion_config.json`, `dl_biocause_graph_pruning_diagnostics.json`), and `reports/` tables and figures.
  - **Pending User Colab Verification:** `DL_03`'s final persistence cell (`id=0b9968aa`, Section 9) recorded a kernel crash during `trainer.save_model()` shard writing (`Writing model shards: 0%| | 0/1`) *after* writing prediction CSVs and `dl_biocause_metrics.json`. SafeTensors persistence (`DL_03` Section 9/10 recovery cell `c94e21b0`) and Joblib `aml_classifier.pkl` + `DL/models/hybrid_biocause/` packaging (`DL_04` Section 11/12 cells `b402e8c3`, `f829a1d7`) are prepared in code and await manual Colab execution and reload verification.

### 2. Notebook Execution Register
| Notebook | Scope | Executed Outputs Status | Persistence / Packaging Status |
| :--- | :--- | :--- | :--- |
| `DL/DL_01_data_preparation.ipynb` | Tokenization, entity marking, BioCause split manifest creation | **Executed** (`33` cells, `13` outputs) | `dl_{corr2cause,biocause}_{train,val,test}.csv` exported |
| `DL/DL_02_corr2cause_distilbert.ipynb` | DistilBERT fine-tuning and evaluation on Corr2Cause | **Executed** (`14` cells, `16` outputs) | `dl_corr2cause_test_predictions.csv` exported |
| `DL/DL_03_biocause_pubmedbert.ipynb` | PubMedBERT fine-tuning, joint epoch/threshold selection, BioCause test | **Eval Executed; Final Save Crashed** (`id=0b9968aa` crashed at shard write after exporting predictions & metrics) | CPU SafeTensors recovery cell (`id=c94e21b0`) pending Colab run |
| `DL/DL_04_hybrid_graph_integration.ipynb` | Validation fusion grid search, frozen test eval, DAG pruning | **Sections 1–9 Executed** (`8` outputs) | Sections 11–12 (`aml_classifier.pkl` & reload check) pending Colab run |
| `DL/DL_05_evaluation_ablation.ipynb` | Global benchmark evaluation, 4D ablations, causal Bayesian reasoning | **Executed** (`13` cells, `38` outputs) | Benchmark/ablation CSVs & PNGs extracted to `reports/` |

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
- **Experimental Progression (AML Baseline $\to$ PubMedBERT $\to$ Hybrid Fusion):**
  - The Phase 1 AML baseline (`Logistic Regression Moderate Weight 1:2`) achieved strong ranking (`ROC-AUC = 0.9431`, `PR-AUC = 0.6119`) and precision (`0.6118` at `0.5000`; `0.6447` at `0.5200`), but limited recall (`0.5200` / `0.4900`, `Pos F1 = 0.5622` / `0.5568`).
  - Fine-tuned PubMedBERT (`checkpoint-3723`, Epoch `3.0`, $\tau^* = 0.0273$) lifted test positive recall to `0.8000` (`TP=80`), but dynamic weighting (`14.9871x`) compressed raw probabilities (`val max = 0.3182`) and raised false positives (`FP=228`), yielding standalone `Pos Precision = 0.2597` and `Pos F1 = 0.3922`.
  - Validation-only fusion ($\alpha^* = 0.45, \tau^* = 0.3200$) combined PubMedBERT's recall with AML's precision, lifting validation `Pos F1` to **`0.6400`** (`Val Macro-F1 = 0.8072`, `+0.0424` over tuned AML) and held-out test `Pos F1` to **`0.5907`** (`Macro-F1 = 0.7801`, `ROC-AUC = 0.9448`, `PR-AUC = 0.6364`). Hybrid selection rests on these observed validation and test gains without claiming statistical significance.
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

## D. Validation & Verification Scope (Static Checks vs. Colab Runtime Execution)

### 1. Static Checks Performed Locally (No Code Executed)
| Validation Check | Scope / Command | Result |
| :--- | :--- | :---: |
| Notebook & JSON Validation | `jq empty` across `DL_01`–`DL_05`, `fusion_config.json`, `model_manifest.json` | **Valid JSON (Exit 0)** |
| Python Syntax & Import Audit | Static inspection of `DL/hybrid_inference.py`, `DL_03`, `DL_04` cells vs. `requirements-dl.txt` | **Verified Statically** |
| Git Whitespace Check | `git diff --check` across modified tracked files | **Clean (Exit 0)** |
| Shared Utilities Preservation | `git hash-object DL/dl_utils.py` | **`1844e06924e6b971c749f4453928c66427ccc9fc` (Exact Match)** |
| Phase 1 Protection | `AML/`, `00_setup.ipynb`, `01_eda.ipynb`, `tests/test_phase1_baseline.py`, `requirements.txt` | **Unmodified** |
| `.gitignore` Binary Policy Check | `models/`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `checkpoints/` | **Confirmed Excluded from Git** |

### 2. Runtime Assertions Observed in Executed Colab Notebook Outputs
| Runtime Check | Executed Notebook Scope | Observed Status |
| :--- | :--- | :---: |
| Candidate ID & Label Alignment | BioCause val ($N=1,884$) & test ($N=1,394$) across AML, DL, and Hybrid (`DL_04`, `DL_05`) | **Passed in Colab Outputs** |
| Probability Bounds $[0, 1]$ | Finite probabilities across all prediction exports (`DL_03`–`DL_05`) | **Passed in Colab Outputs** |
| DAG Acyclicity & Pruning Audit | `nx.is_directed_acyclic_graph(G_dag)` and `86 - 7 = 79` edge reconciliation (`DL_04`, `DL_05`) | **`True` (Passed in Colab Outputs)** |
| PubMedBERT SafeTensors Persistence | `DL_03` Section 9 (`id=0b9968aa`) / Section 10 recovery (`id=c94e21b0`) | **Pending User Colab Execution** *(Cell `0b9968aa` crashed during shard writing)* |
| Hybrid Packaging & Reload Check | `DL_04` Section 11 (`aml_classifier.pkl`, `model_manifest.json`) & Section 12 validation check | **Pending User Colab Execution** |

---

## E. Outstanding Issues and Limitations

1. **PubMedBERT Weight Persistence Crash in Committed `DL_03` Output:** Cell `0b9968aa` (Section 9) successfully exported `dl_biocause_test_predictions.csv` and `dl_biocause_metrics.json`, then suffered a Colab kernel crash at `Writing model shards: 0%| | 0/1`. CPU SafeTensors persistence (`model.to('cpu')`, `safe_serialization=True`) and a standalone recovery cell (`DL_03` Section 10, `id=c94e21b0`) have been prepared without retraining and require manual Colab execution and reload verification.
2. **AML `.pkl` Serialization & Hybrid Package Verification:** Code for `DL/models/hybrid_biocause/aml_classifier.pkl` (`joblib.dump`) and full hybrid reload verification (`DL_04` Sections 11–12, `id=b402e8c3` and `id=f829a1d7`) has been prepared in `DL/hybrid_inference.py` and `DL_04`, and requires manual Colab execution to generate and verify binary weights (not committed due to `.gitignore`).
3. **Synthetic CPDs:** The Bayesian Network demonstrations (`DL_04` Section 8 4-node subnetwork and `DL_05` Section 7 3-node subnetwork) use explicitly synthetic conditional probability distributions (CPDs) to illustrate Pearl's $do$-calculus and must not be interpreted as empirically estimated biological effect sizes.
4. **Low Probability Calibration in Weighted PubMedBERT:** Under dynamic positive class weighting ($14.9871\times$), raw softmax probabilities on validation remain compressed (`max = 0.3182`), requiring validation-selected thresholding ($\tau^* = 0.0273$) rather than a naive $0.50$ cutoff.
5. **Joblib / Pickle Trust Boundary:** `aml_classifier.pkl` uses Joblib serialization and must only be loaded from trusted, project-verified sources (`DL/hybrid_inference.py`).

---

## F. Final Deliverables Register

1. **Notebooks (`DL/`):**
   - `DL/DL_01_data_preparation.ipynb`
   - `DL/DL_02_corr2cause_distilbert.ipynb`
   - `DL/DL_03_biocause_pubmedbert.ipynb` *(Sections 9–10 updated for CPU SafeTensors persistence & recovery)*
   - `DL/DL_04_hybrid_graph_integration.ipynb` *(Sections 11–12 added for `aml_classifier.pkl` & hybrid reload check)*
   - `DL/DL_05_evaluation_ablation.ipynb`
2. **Reconciled Audit, Reports & Documentation (`reports/`, `DL/`, root):**
   - `reports/DL_FINAL_EXECUTION_AUDIT.md`
   - `reports/dl_final_benchmark_summary.csv`
   - `reports/dl_ablation_summary.csv`
   - `reports/dl_biocause_pr_roc_curves.png`
   - `reports/dl_biocause_bayesian_dag_motif.png`
   - `DL/README.md`
   - `README.md`
3. **Deployable Hybrid Packaging & Config (`DL/` & `DL/models/hybrid_biocause/`):**
   - `DL/hybrid_inference.py`
   - `DL/HYBRID_MODEL_PACKAGE.md` *(tracked package, loading & schema documentation)*
   - `DL/models/hybrid_biocause/fusion_config.json` *(gitignored under `models/`)*
   - `DL/models/hybrid_biocause/model_manifest.json` *(gitignored under `models/`)*
   - `DL/models/hybrid_biocause/pubmedbert/` *(`model.safetensors` + `checkpoint_provenance.json`, gitignored; pending Colab verification)*
   - `DL/models/hybrid_biocause/aml_classifier.pkl` *(Joblib pipeline, gitignored; pending Colab verification)*
4. **Pipeline Prediction, Config & Graph Artifacts (Colab-generated in `data/processed/`):**
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
