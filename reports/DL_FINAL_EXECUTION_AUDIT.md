# Final DL Notebook Execution and Independent Audit Report

- **Repository:** `alpha-sml/causal-nlp-reasoning`
- **Branch:** `main`
- **Model & Artifact Verification Baseline Commit:** `4d424f1` (`4d424f115d7c3164c3f5b334a13f61d62470b8d3`)
- **Latest Repository State at Audit Sync:** `e67ffd5` (`e67ffd5551d13b04a31fff832991afe75b3e89c0`)
- **Date of Audit:** `2026-11-17`
- **Status:** EVALUATION, MODEL PERSISTENCE & HYBRID PACKAGING VERIFIED (`reports/dl_report.pdf` SOURCE UNAVAILABLE)

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
- **Execution Mode:** Manual sequential execution of `DL_01` through `DL_05` in Google Colab, combined with static JSON, syntax, `.gitignore`/LFS, and SHA inspection in the local repository (`DL/dl_utils.py` commit `4bf3cddfedadc5c7342c6f5f1f0b04ca7bee7e5a`, blob SHA `1844e06924e6b971c749f4453928c66427ccc9fc`).
- **Executed vs. Static Verification Boundary:**
  - **Executed & Verified in Colab:** Training, joint validation `(epoch, threshold)` selection (`checkpoint-3723`, Epoch `3.0`, $\tau^* = 0.16$), held-out test evaluation, PubMedBERT SafeTensors persistence (`DL_03` Sections 9–10, `model.safetensors = 437,970,936` bytes), Joblib `aml_classifier.pkl` serialization (`6,708` bytes), hybrid reload & validation inference verification (`DL_04` Sections 11–12, `reload_and_inference_verified: true`), and `reports/` CSVs & 4 PNG figures.
  - **Static Verification Performed Locally:** `jq empty`, `git diff --check`, `git hash-object DL/dl_utils.py`, `.gitattributes` Git LFS tracking, and artifact inventory checks (no notebooks, training, or tests were executed locally).

### 2. Notebook Execution Register
| Notebook | Scope | Executed Outputs Status | Persistence / Packaging Status |
| :--- | :--- | :--- | :--- |
| `DL/DL_01_data_preparation.ipynb` | Tokenization, entity marking, BioCause split manifest creation | **Executed (`21/21` code cells)** | `dl_{corr2cause,biocause}_{train,val,test}.csv` generated locally (`data/processed/` gitignored) |
| `DL/DL_02_corr2cause_distilbert.ipynb` | DistilBERT fine-tuning and evaluation on Corr2Cause | **Executed (`9/9` code cells)** | `dl_corr2cause_test_predictions.csv` local (`data/processed/` gitignored); `dl_corr2cause_confusion_matrix.png` in `reports/` |
| `DL/DL_03_biocause_pubmedbert.ipynb` | PubMedBERT fine-tuning, joint epoch/threshold selection, SafeTensors save & verify | **Executed (`10/10` code cells, `0` errors)** | `DL/models/hybrid_biocause/pubmedbert/` (`model.safetensors` via Git LFS) **verified** |
| `DL/DL_04_hybrid_graph_integration.ipynb` | Validation fusion grid search, frozen test eval, DAG pruning, hybrid packaging | **Executed (`11/11` code cells, `0` errors)** | `aml_classifier.pkl`, `fusion_config.json`, `model_manifest.json`, reload check **verified** |
| `DL/DL_05_evaluation_ablation.ipynb` | Global benchmark evaluation, 4D ablations, causal Bayesian reasoning | **Executed (`10/10` code cells, `0` errors)** | Benchmark/ablation CSVs & 4 PNGs committed in `reports/` |

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
- **Validation Selection:** Epoch `3.0` (`validation_loss = 0.3334`, `validation_accuracy = 0.8838`, `validation_precision = 0.7518`, `validation_recall = 0.6361`, `validation_macro_f1 = 0.6688`, `validation_roc_auc = 0.8545`, `validation_pr_auc = 0.4437`).
- **Held-Out Test Performance ($N = 1,162$):**
  - **Accuracy:** $0.8718$ ($1013 / 1162$)
  - **Positive (Class 1 Entailed):** Precision = $0.6476$, Recall = $0.3778$, Pos F1 = $0.4772$
  - **Macro Metrics:** Precision = $0.7708$, Recall = $0.6700$, Macro-F1 = $0.7021$
  - **ROC-AUC:** $0.8794$ | **PR-AUC:** $0.5957$
  - **Confusion Matrix** (`[[TN, FP], [FN, TP]]`):
    $$\begin{pmatrix} 945 & 37 \\ 112 & 68 \end{pmatrix}$$
- **Disaggregated by `num_variables`:**
  | Scope | Examples | TP | FP | FN | TN | Accuracy | Pos Precision | Pos Recall | Pos F1 | Macro F1 | ROC-AUC | PR-AUC |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | `num_variables = 2` | 6 | 0 | 1 | 0 | 5 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.4545 | - | - |
  | `num_variables = 3` | 48 | 3 | 7 | 0 | 38 | 0.8542 | 0.3000 | 1.0000 | 0.4615 | 0.6886 | 0.9778 | 0.8333 |
  | `num_variables = 4` | 72 | 0 | 1 | 0 | 71 | 0.9861 | 0.0000 | 0.0000 | 0.0000 | 0.4965 | - | - |
  | `num_variables = 5` | 514 | 36 | 22 | 21 | 435 | 0.9163 | 0.6207 | 0.6316 | 0.6261 | 0.7895 | 0.9423 | 0.7074 |
  | `num_variables = 6` | 522 | 29 | 6 | 91 | 396 | 0.8142 | 0.8286 | 0.2417 | 0.3742 | 0.6325 | 0.7666 | 0.5497 |
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

- **PubMedBERT Classifier (Joint Validation-Selected `checkpoint-3723`, Epoch `3.0`, $\tau^* = 0.1600$):**
  - **Validation Probability Calibration Summary ($N = 1,884$):** min = $0.0009$, median = $0.0067$, max = $0.5028$.
  - **Validation Metrics at $\tau^* = 0.1600$:** `178` positive predictions, Pos Precision = $0.3933$, Pos Recall = $0.5344$, Pos F1 = $0.4531$, Macro-F1 = $0.7021$, Validation Confusion Matrix (`[[TN, FP], [FN, TP]]`): `[[1645, 108], [61, 70]]`.
  - **Held-Out Test at Frozen (`checkpoint-3723`, $\tau^* = 0.1600$):** Accuracy = $0.8996$ ($1254 / 1394$), Pos Precision = $0.3718$, Pos Recall = $0.5800$, Pos F1 = $0.4531$, Macro-F1 = $0.6989$, ROC-AUC = $0.8950$, PR-AUC = $0.3744$.
  - **Test Confusion Matrix** (`[[TN, FP], [FN, TP]]`): `[[1196, 98], [42, 58]]`

### 4. Hybrid Integration, Graph Refinement & Illustrative Do-Calculus (`DL_04` & `DL_05`)
- **Experimental Progression (AML Baseline $\to$ PubMedBERT $\to$ Hybrid Fusion):**
  - The Phase 1 AML baseline (`Logistic Regression Moderate Weight 1:2`) achieved strong ranking (`ROC-AUC = 0.9431`, `PR-AUC = 0.6119`) and precision (`0.6118` at `0.5000`; `0.6447` at `0.5200`), with moderate recall (`0.5200` / `0.4900`, `Pos F1 = 0.5622` / `0.5568`).
  - Fine-tuned PubMedBERT (`checkpoint-3723`, Epoch `3.0`, $\tau^* = 0.1600$) lifted test positive recall to `0.5800` (`TP=58`), but dynamic weighting (`14.9871x`) compressed raw probabilities (`val max = 0.5028`) and raised false positives (`FP=98`), yielding standalone `Pos Precision = 0.3718` and `Pos F1 = 0.4531`.
  - Validation-only fusion ($\alpha^* = 0.25, \tau^* = 0.4400$) combined PubMedBERT's contextual signal with AML's precision, lifting validation `Pos F1` to **`0.6383`** (`Val Macro-F1 = 0.8071`, `+0.0407` over tuned AML) and achieving the highest held-out test ranking metrics (**`ROC-AUC = 0.9478`**, **`PR-AUC = 0.6269`**) while cutting PubMedBERT's false positives by `69%` (`98 -> 30`, `Pos F1 = 0.5475`, `Macro-F1 = 0.7582`). Hybrid selection rests on these observed validation gains without claiming statistical significance.
- **Fusion Formula:**
  $$P_{\text{hybrid}}^{(i)} = \alpha \cdot P_{\text{DL}}^{(i)} + (1 - \alpha) \cdot P_{\text{AML}}^{(i)}$$
- **Validation Parameter Selection ($N = 1,884$):**
  - Grid search over $\alpha \in [0.00, 1.00]$ (step $0.05$) and candidate thresholds $\tau \in [0.01, 0.99]$ + validation quantiles with positive-F1-first objective.
  - Optimal validation configuration: $\alpha^* = 0.25$, $\tau^* = 0.4400$.
  - Validation metrics at $(\alpha^* = 0.25, \tau^* = 0.4400)$: Pos F1 = **$0.6383$**, Macro-F1 = **$0.8071$** ($+0.0407$ validation Pos F1 over tuned AML).
- **Held-Out Test Evaluation ($N = 1,394$, Parameters Frozen at $\alpha^* = 0.25, \tau^* = 0.4400$):**
  - **Accuracy:** $0.9419$ ($1313 / 1394$)
  - **Pos Precision:** $0.6203$ | **Pos Recall:** $0.4900$
  - **Pos F1:** $0.5475$ | **Macro-F1:** $0.7582$ | **ROC-AUC:** **$0.9478$** | **PR-AUC:** **$0.6269$**
  - **Test Confusion Matrix** (`[[TN, FP], [FN, TP]]`): `[[1264, 30], [51, 49]]`
- **Post-Processing & Graph Refinement:**
  1. **Linguistic Sanity Filter:** `79` initial positive predictions $\to$ `6` rejected by negation (`0` by distance, `0` both) $\to$ `73` surviving positive relations (Test Accuracy = $0.9419$, Pos Precision = $0.6301$, Pos Recall = $0.4600$, Pos F1 = $0.5318$, Macro-F1 = $0.7504$, Confusion Matrix `[[1267, 27], [54, 46]]`).
  2. **Deterministic Cycle Breaking (`enforce_dag_acyclicity`):** `73` raw edges (`Is DAG = False`) $\to$ pruned `4` bidirectional conflict edges (`0` self-loops, `0` cycle feedback edges) $\to$ `69` acyclic directed edges (`Is DAG = True`).
  3. **Graph Edge Evaluation ($100$ Gold Test Relations):**
     - **Raw Extracted Graph (Pre-Pruning):** `73` edges, `TP = 46, FP = 27, FN = 54`, Precision = $0.6301$, Recall = $0.4600$, F1 = $0.5318$, `Is DAG = False`.
     - **Acyclic Causal DAG (Post-Pruning):** `69` edges, `TP = 46, FP = 23, FN = 54`, Precision = **$0.6667$**, Recall = $0.4600$, F1 = **$0.5444$**, `Is DAG = True`. *(All 4 bidirectional conflict removals eliminated false positives while retaining all 46 true positives.)*
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
| `Full model with markers and weighted loss` | `completed` | `checkpoint-3723` | `3.0` | `0.1600` | 0.4531 | 0.7021 | 58 | 98 | 42 | 1196 | 0.8996 | 0.3718 | **0.5800** | 0.4531 | 0.6989 | 0.8950 | 0.3744 |
| `No entity markers` | `completed` | `checkpoint-2482` | `2.0` | `0.0593` | 0.2312 | 0.5506 | 60 | 244 | 40 | 1050 | 0.7963 | 0.1974 | 0.6000 | 0.2970 | 0.5890 | 0.7980 | 0.1941 |
| `Entity markers and unweighted loss` | `completed` | `checkpoint-3723` | `3.0` | `0.0800` | 0.4558 | 0.7049 | 55 | 89 | 45 | 1205 | 0.9039 | 0.3819 | 0.5500 | 0.4508 | 0.6991 | 0.8994 | 0.3743 |
| `Pure AML (fixed 0.50)` | `completed` | `-` | `-` | `0.5000` | - | - | 52 | 33 | 48 | 1261 | 0.9419 | 0.6118 | 0.5200 | **0.5622** | **0.7655** | 0.9431 | 0.6119 |
| `AML-only validation-selected threshold` | `completed` | `-` | `-` | `0.5200` | 0.5976 | 0.7844 | 49 | 27 | 51 | 1267 | **0.9440** | **0.6447** | 0.4900 | 0.5568 | 0.7635 | 0.9431 | 0.6119 |
| `Validation-selected hybrid` | `completed` | `-` | `-` | `0.4400` | **0.6383** | **0.8071** | 49 | 30 | 51 | 1264 | 0.9419 | 0.6203 | 0.4900 | 0.5475 | 0.7582 | **0.9478** | **0.6269** |
| `Sanitized hybrid (linguistic filtering)` | `completed` | `-` | `-` | `0.4400` | - | - | 46 | 27 | 54 | 1267 | 0.9419 | 0.6301 | 0.4600 | 0.5318 | 0.7504 | **0.9478** | **0.6269** |

Key ablation findings:
1. **Entity Markers (`[E1]`, `[E2]`):** Removing explicit argument markers degrades test Pos F1 from `0.4531` to `0.2970` (`-0.1561`), ROC-AUC from `0.8950` to `0.7980`, and PR-AUC from `0.3744` to `0.1941`.
2. **Dynamic Class Weighting vs. Calibrated Thresholding:** At fixed `0.50` (`argmax`), unweighted cross-entropy predicts `0` positives across all 3 epochs. Under joint validation epoch and threshold calibration (`checkpoint-3723`, Epoch `3.0`, $\tau = 0.0800$), unweighted CE reaches slightly higher standalone precision (`0.3819` vs `0.3718`), while dynamic weighting ($w_{\text{pos}} = 14.9871$) achieves higher positive recall (`0.5800` vs `0.5500`), Pos F1 (`0.4531` vs `0.4508`), and PR-AUC (`0.3744` vs `0.3743`).
3. **Hybrid Ensembling:** Validation-tuned linear fusion ($\alpha^* = 0.25, \tau^* = 0.4400$) outperforms every standalone model on validation (`Val Pos F1 = 0.6383`, `Val Macro-F1 = 0.8071`) and achieves the highest held-out test `ROC-AUC` (`0.9478`) and `PR-AUC` (`0.6269`), while cutting PubMedBERT's false positives by `69%` (`98 -> 30`).
4. **Linguistic Post-Processing & DAG Constraints:** Linguistic negation filtering removes 3 TPs alongside 3 FPs at candidate level (`Pos F1: 0.5475 -> 0.5318`), while subsequent `enforce_dag_acyclicity` pruning eliminates 4 strictly false-positive bidirectional conflicts, boosting graph edge Precision to `0.6667` and Graph F1 to `0.5444`.

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
| `.gitignore` & Git LFS Policy Check | `DL/models/hybrid_biocause/**` unignored; `model.safetensors` tracked via `.gitattributes` (Git LFS); `checkpoints/` & `data/processed/*` ignored | **Verified via `git check-ignore -v` & `git ls-tree`** |

### 2. Runtime Assertions Observed in Executed Colab Notebook Outputs
| Runtime Check | Executed Notebook Scope | Observed Status |
| :--- | :--- | :---: |
| Candidate ID & Label Alignment | BioCause val ($N=1,884$) & test ($N=1,394$) across AML, DL, and Hybrid (`DL_04`, `DL_05`) | **Passed in Colab Outputs** |
| Probability Bounds $[0, 1]$ | Finite probabilities across all prediction exports (`DL_03`–`DL_05`) | **Passed in Colab Outputs** |
| DAG Acyclicity & Pruning Audit | `nx.is_directed_acyclic_graph(G_dag)` and `73 - 4 = 69` edge reconciliation (`DL_04`, `DL_05`) | **`True` (Passed in Colab Outputs)** |
| PubMedBERT SafeTensors Persistence | `DL_03` Section 9 (`id=0b9968aa`) & Section 10 recovery (`id=c94e21b0`) | **Verified (`model.safetensors = 437,970,936` bytes)** |
| Hybrid Packaging & Reload Check | `DL_04` Section 11 (`aml_classifier.pkl`, `model_manifest.json`) & Section 12 validation reload check | **Verified (`reload_and_inference_verified: true`; `val_max_abs_diff = 1.11e-16`)** |

---

## E. Outstanding Issues and Limitations

1. **Synthetic CPDs:** The Bayesian Network demonstrations (`DL_04` Section 8 4-node subnetwork and `DL_05` Section 7 3-node subnetwork) use explicitly synthetic conditional probability distributions (CPDs) to illustrate Pearl's $do$-calculus and must not be interpreted as empirically estimated biological effect sizes.
2. **Low Probability Calibration in Weighted PubMedBERT:** Under dynamic positive class weighting ($14.9871\times$), raw softmax probabilities on validation remain compressed (`max = 0.5028`), requiring validation-selected thresholding ($\tau^* = 0.1600$) rather than a naive $0.50$ cutoff.
3. **Joblib / Pickle Trust Boundary:** `aml_classifier.pkl` uses Joblib serialization and must only be loaded from trusted, project-verified sources (`DL/hybrid_inference.py`).
4. **Git LFS Requirement for Cloning Weights:** `DL/models/hybrid_biocause/pubmedbert/model.safetensors` (`417.68 MB`) is stored via Git LFS (`.gitattributes`), so cloning environments must have `git-lfs` installed to pull the full SafeTensors weights rather than the LFS pointer.
5. **Missing Editable Source for `reports/dl_report.pdf`:** No editable source file (`.tex`, `.docx`, `.qmd`, `.typ`, `.rmd`) for `reports/dl_report.pdf` exists in the repository, so `reports/dl_report.pdf` was not fabricated or modified. Consequently, `reports/dl_report.pdf` (Page 1 Abstract, Page 2 Section V.A, Page 4 Section VII.A & Fig. 6, Page 5 Table II, Page 6 Section X) still reflects the earlier DistilBERT run (`0.8623` test accuracy, `0.6791` macro-F1, `0.4366` pos F1) rather than the latest `DL_02`/`DL_05` results (`0.8718` accuracy, `0.7021` macro-F1, `0.4772` pos F1); updating it requires supplying the original editable document source.

---

## F. Final Deliverables Register

1. **Committed Notebooks (`DL/`):**
   - `DL/DL_01_data_preparation.ipynb`
   - `DL/DL_02_corr2cause_distilbert.ipynb`
   - `DL/DL_03_biocause_pubmedbert.ipynb` *(Sections 9–10 CPU SafeTensors persistence & verification executed)*
   - `DL/DL_04_hybrid_graph_integration.ipynb` *(Sections 11–12 `aml_classifier.pkl` & hybrid reload verification executed)*
   - `DL/DL_05_evaluation_ablation.ipynb`
2. **Committed Audit, Reports & Visuals (`reports/`, `DL/`, root):**
   - `reports/DL_FINAL_EXECUTION_AUDIT.md`
   - `reports/dl_final_benchmark_summary.csv`
   - `reports/dl_ablation_summary.csv`
   - `reports/dl_biocause_pr_roc_curves.png`
   - `reports/dl_biocause_model_comparison.png`
   - `reports/dl_biocause_confusion_matrices.png`
   - `reports/dl_corr2cause_confusion_matrix.png`
   - `reports/dl_biocause_bayesian_dag_motif.png`
   - `reports/dl_report.pdf` & `reports/dl_phase.pptx`
   - `DL/README.md`
   - `README.md`
3. **Committed Deployable Hybrid Package (`DL/`, `DL/models/hybrid_biocause/`, `.gitattributes`):**
   - `DL/hybrid_inference.py`
   - `DL/HYBRID_MODEL_PACKAGE.md` *(tracked package, loading & schema documentation)*
   - `.gitattributes` *(Git LFS tracking for `DL/models/hybrid_biocause/pubmedbert/model.safetensors`)*
   - `DL/models/hybrid_biocause/fusion_config.json`
   - `DL/models/hybrid_biocause/model_manifest.json`
   - `DL/models/hybrid_biocause/aml_classifier.pkl` (`6,708` bytes)
   - `DL/models/hybrid_biocause/pubmedbert/model.safetensors` (`437,970,936` bytes, Git LFS)
   - `DL/models/hybrid_biocause/pubmedbert/config.json`
   - `DL/models/hybrid_biocause/pubmedbert/checkpoint_provenance.json`
   - `DL/models/hybrid_biocause/pubmedbert/tokenizer.json`
   - `DL/models/hybrid_biocause/pubmedbert/tokenizer_config.json`
4. **Local-Only Pipeline Prediction, Config & Graph Artifacts (`data/processed/` — Intentionally Excluded by `.gitignore`):**
   Only `data/processed/.gitkeep` is committed; the following Colab-generated files remain local (`data/processed/*` in `.gitignore`) and are reproducible by running `DL_01`–`DL_04`:
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
