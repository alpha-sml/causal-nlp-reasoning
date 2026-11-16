# Deployable Hybrid BioCause Model Package (`DL/models/hybrid_biocause/`)

This tracked guide documents how the Phase 2 hybrid biomedical causal-relation extraction package is structured, serialized, verified, and loaded for inference using [`DL/hybrid_inference.py`](hybrid_inference.py).

---

## 1. Why the Hybrid Package Was Selected

Across identical held-out BioCause test candidates (`N = 1,394`; `100` positive, `1,294` negative) and validation candidates (`N = 1,884`; `131` positive, `1,753` negative):
1. **Phase 1 AML Baseline (`Logistic Regression (Moderate Weight 1:2)`)**: Delivers strong precision and ranking (`ROC-AUC = 0.9431`, `PR-AUC = 0.6119`), achieving `Pos F1 = 0.5622` (`P=0.6118, R=0.5200`) at fixed `0.5000` and `Pos F1 = 0.5568` (`P=0.6447, R=0.4900`) at validation-selected `0.5200` (`Val Pos F1 = 0.5976`).
2. **Phase 2 PubMedBERT (`checkpoint-3723`, Epoch `3.0`, $\tau^* = 0.0273$)**: Captures broader contextual signals and reaches the highest positive recall (`0.8000` test, `TP=80`), but class weighting (`9307 / 621 = 14.9871`) compresses raw probabilities (`val max = 0.3182`) and increases false positives (`FP=228`), lowering standalone `Pos Precision` (`0.2597`) and `Pos F1` (`0.3922`, `Val Pos F1 = 0.3701`).
3. **Validation-Selected Hybrid Fusion ($\alpha^* = 0.45, \tau^* = 0.3200$)**: Combining $0.45 \cdot P_{\text{DL}} + 0.55 \cdot P_{\text{AML}}$ pairs PubMedBERT's recall with AML's high-precision structural signal, lifting validation `Pos F1` to **`0.6400`** (`Val Macro-F1 = 0.8072`, `+0.0424` over tuned AML) and held-out test `Pos F1` to **`0.5907`** (`Macro-F1 = 0.7801`, `ROC-AUC = 0.9448`, `PR-AUC = 0.6364`). No statistical significance claim is made; selection follows the observed validation and test gains.

---

## 2. Package Layout & `.gitignore` Storage Policy

```text
DL/models/hybrid_biocause/
├── pubmedbert/                     # Generated in Colab: SafeTensors weights + tokenizer
│   ├── config.json
│   ├── model.safetensors           # Required SafeTensors format (no .bin fallback)
│   ├── checkpoint_provenance.json  # Records selected_checkpoint_name, step, epoch, threshold
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
├── aml_classifier.pkl              # Generated in Colab: Joblib-serialized sklearn Pipeline
├── fusion_config.json              # Frozen fusion parameters (alpha_star=0.45, threshold_star=0.32)
├── model_manifest.json             # Provenance, split counts, and persistence verification status
└── README.md                       # Local package copy (untracked under .gitignore)
```

Per `.gitignore` (`models/`, `checkpoints/`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`), `DL/models/` is excluded from Git tracking so large neural weights and generated binaries are never committed to Git history. All code required to reproduce and verify the package is tracked in [`DL/hybrid_inference.py`](hybrid_inference.py), [`DL/DL_03_biocause_pubmedbert.ipynb`](DL_03_biocause_pubmedbert.ipynb), and [`DL/DL_04_hybrid_graph_integration.ipynb`](DL_04_hybrid_graph_integration.ipynb).

---

## 3. Colab Generation & Verification Workflow

1. **Persist & Verify PubMedBERT (`pubmedbert/model.safetensors`)**:
   - In [`DL/DL_03_biocause_pubmedbert.ipynb`](DL_03_biocause_pubmedbert.ipynb), execute **Section 9** (`id="0b9968aa"`) or standalone **Section 10** (`id="c94e21b0"`) against the recorded frozen checkpoint (`checkpoint-3723`).
   - Offloads the model to CPU, writes `model.safetensors` (`safe_serialization=True` required; no `.bin` fallback), writes `checkpoint_provenance.json`, reloads from disk, verifies `num_labels == 2` and marker vocabulary alignment, and runs a 4-sample validation inference check without test labels.
2. **Serialize `aml_classifier.pkl` & Package `hybrid_biocause/`**:
   - In [`DL/DL_04_hybrid_graph_integration.ipynb`](DL_04_hybrid_graph_integration.ipynb), execute **Section 11** (`id="b402e8c3"`) and **Section 12** (`id="f829a1d7"`).
   - Section 11 fits `BioCauseAMLFeatureExtractor` + `LogisticRegression(class_weight={0: 1.0, 1: 2.0}, C=1.0, max_iter=200, random_state=42)` strictly on `split == 'train'` (`10,086` rows), verifies `atol=1e-6` agreement against `aml_biocause_validation_predictions.csv` and `aml_biocause_test_predictions.csv`, requires `dl_biocause_metrics.json`, verifies `checkpoint_provenance.json` + `model.safetensors` match the recorded selected checkpoint (never using stale weights or default checkpoint names), dumps `aml_classifier.pkl` via `joblib.dump`, and writes `fusion_config.json` and `model_manifest.json`.
   - Section 12 reloads all components and verifies deterministic inference on an unlabeled 8-row validation slice.

---

## 4. Loading & Inference Usage

> [!CAUTION]
> **Pickle / Joblib Security**: `aml_classifier.pkl` uses Joblib serialization, which can execute arbitrary Python code on untrusted inputs. Load `aml_classifier.pkl` **only** from trusted, project-verified outputs. PubMedBERT weights are stored strictly in SafeTensors (`model.safetensors`), never pickled.

```python
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from DL.hybrid_inference import HybridBioCausePredictor

# Option A: Unified hybrid inference wrapper (AML + PubMedBERT + frozen fusion + linguistic filter + DAG)
predictor = HybridBioCausePredictor("DL/models/hybrid_biocause")
df_preds = predictor.predict_candidates(df_biocause_candidates)
df_relations, G_dag, audit_log = predictor.build_causal_dag(
    df_biocause_candidates, df_preds, split_name="inference", use_sanitized=True
)

# Option B: Direct component loading (ensure DL/ is on sys.path before joblib.load)
aml_pipeline = joblib.load("DL/models/hybrid_biocause/aml_classifier.pkl")
tokenizer = AutoTokenizer.from_pretrained("DL/models/hybrid_biocause/pubmedbert", local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "DL/models/hybrid_biocause/pubmedbert", local_files_only=True
)
```

---

## 5. Input & Output Schemas

- **Required Input Columns**: `candidate_id`, `char_distance`, `arg1_before_arg2`, `between_text`, `arg1_text`, `arg2_text`, plus either `marked_text` (containing `[E1]`, `[/E1]`, `[E2]`, `[/E2]`) or standoff columns (`section_id`, `arg1_start`, `arg1_end`, `arg2_start`, `arg2_end`). Corr2Cause columns (`premise`, `hypothesis`) are explicitly rejected.
- **Prediction Output Columns**: `candidate_id`, `aml_prob`, `dl_prob`, `hybrid_prob`, `hybrid_pred` (`>= 0.32`), `flag_negation`, `flag_distance`, `sanitized_pred`.
- **Relation Output**: 17-column `dl_utils.UNIFIED_RELATION_SCHEMA` and acyclic `networkx.DiGraph` via `enforce_dag_acyclicity`.
