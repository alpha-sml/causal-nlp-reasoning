"""hybrid_inference.py

Reusable persistence, packaging, and inference utilities for the Phase 2
Hybrid BioCause Causal Relation Extraction pipeline.

Combines:
1. Phase 1 AML Pipeline (`Logistic Regression (Moderate Weight 1:2)` with exact
   10 dense structural/lexical features + StandardScaler + 100-feature TF-IDF)
   serialized safely with `joblib`.
2. Phase 2 PubMedBERT Fulltext Classifier with entity markers ([E1]..[/E2])
   persisted in HuggingFace SafeTensors/tokenizer format (never pickled).
3. Frozen validation-selected fusion configuration (`alpha_star`, `threshold_star`)
   and downstream linguistic filtering / 17-column UnifiedRelationRecord / DAG
   compilation aligned with DL_03 and DL_04.

Security Note:
    `aml_classifier.pkl` uses Joblib/pickle serialization for the scikit-learn
    pipeline. Load `.pkl` artifacts only from trusted, project-verified sources.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_DL_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _DL_DIR.parent
_AML_DIR = _ROOT_DIR / "AML"
for _path in (_ROOT_DIR, _AML_DIR, _DL_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import dl_utils  # noqa: E402
from causal_extractors import NEGATION_PATTERN, enforce_dag_acyclicity  # noqa: E402


PAT_ACTIVE = re.compile(
    r"(?i)\b(?:causes|promotes|induces|activates|leads to|results in)\b"
)
PAT_PASSIVE = re.compile(
    r"(?i)\b(?:is caused by|results from|due to|is mediated by)\b"
)
PAT_DISCOURSE = re.compile(
    r"(?i)\b(?:suggesting that|indicating that|because|therefore|thus)\b"
)
PAT_NEGATION = re.compile(r"(?i)\b(?:not|never|no|failed|without)\b")
PAT_UNCERTAINTY = re.compile(
    r"(?i)\b(?:may|might|could|suggests|possible)\b"
)

DENSE_FEATURE_NAMES: Tuple[str, ...] = (
    "char_distance",
    "arg1_before_arg2",
    "between_char_len",
    "arg1_char_len",
    "arg2_char_len",
    "has_active_causal_cue",
    "has_passive_causal_cue",
    "has_discourse_cue",
    "has_negation_cue",
    "has_uncertainty_cue",
)

REQUIRED_AML_INPUT_COLUMNS: Tuple[str, ...] = (
    "candidate_id",
    "char_distance",
    "arg1_before_arg2",
    "between_text",
    "arg1_text",
    "arg2_text",
)


def extract_aml_dense_features(df: pd.DataFrame) -> np.ndarray:
    """Compute the 10 Phase 1 structural and lexical indicator features."""
    rows: List[List[float]] = []
    for _, r in df.iterrows():
        between_val = r["between_text"]
        b_txt = "" if pd.isna(between_val) else str(between_val)
        arg1_val = r["arg1_text"]
        arg2_val = r["arg2_text"]
        rows.append(
            [
                float(r["char_distance"]),
                float(r["arg1_before_arg2"]),
                float(len(b_txt)),
                float(len("" if pd.isna(arg1_val) else str(arg1_val))),
                float(len("" if pd.isna(arg2_val) else str(arg2_val))),
                float(bool(PAT_ACTIVE.search(b_txt))),
                float(bool(PAT_PASSIVE.search(b_txt))),
                float(bool(PAT_DISCOURSE.search(b_txt))),
                float(bool(PAT_NEGATION.search(b_txt))),
                float(bool(PAT_UNCERTAINTY.search(b_txt))),
            ]
        )
    return np.array(rows, dtype=np.float32)


def _validate_aml_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required BioCause candidate columns before AML transformation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame input, got {type(df).__name__}.")
    if "premise" in df.columns or "hypothesis" in df.columns:
        raise ValueError(
            "Corr2Cause inputs (premise/hypothesis) must never be fed into the "
            "BioCause hybrid inference pipeline."
        )
    missing = [col for col in REQUIRED_AML_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required BioCause AML columns: {missing}")
    if df.empty:
        raise ValueError("Input BioCause candidate DataFrame is empty.")
    return df


class BioCauseAMLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Exact Phase 1 AML feature transformer (10 scaled dense + 100 TF-IDF features).

    Preserves the exact feature engineering, scaling, and TF-IDF n-gram behavior
    defined in `AML/AML_04_classical_classifier.ipynb` and exported in
    `AML/AML_05_integration_evaluation.ipynb`.
    """

    def __init__(self, max_features: int = 100, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "BioCauseAMLFeatureExtractor":
        df = _validate_aml_input_df(X)
        X_dense = extract_aml_dense_features(df)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_dense)
        self.tfidf_ = TfidfVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range
        )
        between_texts = df["between_text"].fillna("").astype(str)
        self.tfidf_.fit(between_texts)
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        if not hasattr(self, "scaler_") or not hasattr(self, "tfidf_"):
            raise RuntimeError("BioCauseAMLFeatureExtractor must be fitted before transform().")
        df = _validate_aml_input_df(X)
        X_dense = extract_aml_dense_features(df)
        X_dense_scaled = self.scaler_.transform(X_dense)
        between_texts = df["between_text"].fillna("").astype(str)
        X_tfidf = self.tfidf_.transform(between_texts)
        return hstack([X_dense_scaled, X_tfidf])

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        if not hasattr(self, "tfidf_"):
            raise RuntimeError("BioCauseAMLFeatureExtractor must be fitted first.")
        tfidf_names = [f"tfidf_{name}" for name in self.tfidf_.get_feature_names_out()]
        return np.array(list(DENSE_FEATURE_NAMES) + tfidf_names, dtype=object)


def build_aml_pipeline() -> Pipeline:
    """Build the unfitted designated Phase 1 AML baseline pipeline."""
    return Pipeline(
        [
            ("features", BioCauseAMLFeatureExtractor(max_features=100, ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(
                    class_weight={0: 1.0, 1: 2.0},
                    C=1.0,
                    max_iter=200,
                    random_state=42,
                ),
            ),
        ]
    )


def fit_and_serialize_aml_pipeline(
    candidates_csv_path: Union[str, Path],
    output_pkl_path: Union[str, Path],
) -> Pipeline:
    """Fit `Logistic Regression (Moderate Weight 1:2)` strictly on train and dump via Joblib."""
    cand_path = Path(candidates_csv_path)
    if not cand_path.is_file():
        raise FileNotFoundError(f"BioCause candidates CSV not found: {cand_path}")
    df_candidates = pd.read_csv(cand_path, keep_default_na=False)
    if "split" not in df_candidates.columns or "label" not in df_candidates.columns:
        raise ValueError(f"Candidates CSV {cand_path} must contain 'split' and 'label' columns.")

    df_train = df_candidates[df_candidates["split"] == "train"].copy()
    if df_train.empty:
        raise ValueError("No rows with split == 'train' found in BioCause candidates CSV.")

    aml_pipeline = build_aml_pipeline()
    aml_pipeline.fit(df_train, df_train["label"].astype(int).to_numpy())

    out_path = Path(output_pkl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(aml_pipeline, out_path)

    reloaded = load_aml_pipeline(out_path)
    if not hasattr(reloaded, "predict_proba") or not hasattr(reloaded, "predict"):
        raise RuntimeError(f"Reloaded AML pipeline from {out_path} lacks predict_proba/predict.")
    return reloaded


def load_aml_pipeline(pkl_path: Union[str, Path]) -> Pipeline:
    """Load `aml_classifier.pkl` using Joblib.

    Security Note:
        Joblib/pickle deserialization can execute arbitrary Python code.
        Load `.pkl` files strictly from trusted, verified project outputs.
    """
    dl_dir_str = str(_DL_DIR)
    if dl_dir_str not in sys.path:
        sys.path.insert(0, dl_dir_str)
    path = Path(pkl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Serialized AML pipeline not found: {path}")
    return joblib.load(path)


def verify_aml_pipeline_equivalence(
    aml_pipeline: Pipeline,
    candidates_csv_path: Union[str, Path],
    val_pred_csv_path: Union[str, Path],
    test_pred_csv_path: Union[str, Path],
    atol: float = 1e-6,
) -> Dict[str, float]:
    """Verify serialized AML pipeline matches exported validation and test `aml_prob`."""
    df_cands_all = pd.read_csv(candidates_csv_path, keep_default_na=False).set_index("candidate_id")
    df_val_ref = pd.read_csv(val_pred_csv_path)
    df_test_ref = pd.read_csv(test_pred_csv_path)

    val_inputs = df_cands_all.loc[df_val_ref["candidate_id"]].reset_index()
    test_inputs = df_cands_all.loc[df_test_ref["candidate_id"]].reset_index()

    val_probs = np.asarray(aml_pipeline.predict_proba(val_inputs)[:, 1], dtype=np.float64)
    test_probs = np.asarray(aml_pipeline.predict_proba(test_inputs)[:, 1], dtype=np.float64)

    val_max_diff = float(np.max(np.abs(val_probs - df_val_ref["aml_prob"].to_numpy())))
    test_max_diff = float(np.max(np.abs(test_probs - df_test_ref["aml_prob"].to_numpy())))

    if val_max_diff > atol:
        raise AssertionError(
            f"AML pipeline validation probability mismatch (max_diff={val_max_diff:.2e} > {atol})"
        )
    if test_max_diff > atol:
        raise AssertionError(
            f"AML pipeline test probability mismatch (max_diff={test_max_diff:.2e} > {atol})"
        )
    return {"val_max_abs_diff": val_max_diff, "test_max_abs_diff": test_max_diff}


def collect_artifact_inventory(directory: Union[str, Path]) -> List[Dict[str, Any]]:
    """Collect relative filenames and byte sizes for saved model artifacts."""
    root = Path(directory)
    if not root.exists():
        return []
    records: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            records.append(
                {
                    "file": str(p.relative_to(root)),
                    "size_bytes": int(p.stat().st_size),
                    "size_mb": round(float(p.stat().st_size) / (1024.0 * 1024.0), 4),
                }
            )
    return records


def persist_and_verify_pubmedbert(
    source_checkpoint_dir: Union[str, Path],
    output_dirs: Sequence[Union[str, Path]],
    sample_texts: Optional[Sequence[str]] = None,
    max_length: int = 384,
    provenance: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Save a frozen PubMedBERT checkpoint strictly as SafeTensors, reload, and verify.

    Moves the model cleanly to CPU before serialization to avoid GPU OOM / kernel crashes
    during shard writing, requires `model.safetensors` (`safe_serialization=True`),
    records `checkpoint_provenance.json`, reloads from disk, verifies architecture,
    vocabulary, and head dimensions, and runs a deterministic inference sanity check
    without test labels.
    """
    import gc
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt_dir = Path(source_checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Frozen PubMedBERT checkpoint directory not found: {ckpt_dir}. "
            "Do not substitute an untrained base checkpoint."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    missing_markers = [
        m
        for m in dl_utils.ENTITY_MARKERS
        if tokenizer.convert_tokens_to_ids(m) in (None, tokenizer.unk_token_id)
    ]
    if missing_markers:
        raise ValueError(f"Checkpoint tokenizer missing entity markers: {missing_markers}")

    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir), local_files_only=True
    )
    model.to("cpu")
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prov_payload: Dict[str, Any] = {
        "selected_checkpoint": str(ckpt_dir),
        "selected_checkpoint_name": ckpt_dir.name,
        "weight_format": "safetensors",
    }
    if provenance:
        prov_payload.update(provenance)

    inventory_all: List[Dict[str, Any]] = []
    for out_dir_raw in output_dirs:
        out_dir = Path(out_dir_raw)
        out_dir.mkdir(parents=True, exist_ok=True)
        stale_bin = out_dir / "pytorch_model.bin"
        if stale_bin.exists():
            stale_bin.unlink()

        model.save_pretrained(str(out_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(out_dir))
        (out_dir / "checkpoint_provenance.json").write_text(
            json.dumps(prov_payload, indent=2) + "\n"
        )

        assert (out_dir / "config.json").is_file(), f"Missing config.json in {out_dir}"
        assert (out_dir / "model.safetensors").is_file(), (
            f"Required SafeTensors weights file missing in {out_dir}"
        )
        has_tok = (out_dir / "tokenizer_config.json").is_file() and (
            (out_dir / "vocab.txt").is_file() or (out_dir / "tokenizer.json").is_file()
        )
        assert has_tok, f"Missing tokenizer artifacts in {out_dir}"

        reloaded_tok = AutoTokenizer.from_pretrained(str(out_dir), local_files_only=True)
        reloaded_model = AutoModelForSequenceClassification.from_pretrained(
            str(out_dir), local_files_only=True
        )
        reloaded_model.eval()

        assert reloaded_model.config.num_labels == 2, "Expected binary classification head (num_labels=2)"
        assert (
            reloaded_model.get_input_embeddings().weight.shape[0] == len(reloaded_tok)
        ), "Embedding size mismatch against reloaded tokenizer vocabulary"
        for marker in dl_utils.ENTITY_MARKERS:
            tid = reloaded_tok.convert_tokens_to_ids(marker)
            assert tid is not None and tid != reloaded_tok.unk_token_id, (
                f"Reloaded tokenizer missing special marker {marker}"
            )
        cls_head = (
            reloaded_model.classifier.out_proj
            if hasattr(reloaded_model.classifier, "out_proj")
            else reloaded_model.classifier
        )
        assert cls_head.out_features == 2, f"Expected out_features=2, got {cls_head.out_features}"
        assert torch.isfinite(cls_head.weight).all() and torch.isfinite(cls_head.bias).all()

        if sample_texts:
            encoded = reloaded_tok(
                list(sample_texts),
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = reloaded_model(**encoded).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            assert probs.shape == (len(sample_texts),)
            assert np.isfinite(probs).all()
            assert ((probs >= 0.0) & (probs <= 1.0)).all()

        for rec in collect_artifact_inventory(out_dir):
            inventory_all.append({"directory": str(out_dir.name), **rec})

        del reloaded_model, reloaded_tok

    del model, tokenizer
    gc.collect()
    return inventory_all


def logits_to_positive_probabilities(raw_logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax returning class-1 probabilities in [0, 1]."""
    logits = np.asarray(raw_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected 2-class logits shape (N, 2), got {logits.shape}.")
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits encountered during PubMedBERT inference.")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    return probs[:, 1]


class HybridBioCausePredictor:
    """End-to-end reusable inference wrapper for the packaged hybrid BioCause model.

    Loads:
        - `pubmedbert/` via HuggingFace `from_pretrained(..., local_files_only=True)`
        - `aml_classifier.pkl` via `joblib.load`
        - `fusion_config.json` containing frozen `alpha_star` and `threshold_star`

    Enforces:
        - Strict separation from Corr2Cause inputs.
        - Exact candidate_id identity and row ordering across AML and DL streams.
        - Zero threshold or alpha tuning at inference time.
        - No access to test labels during inference.
    """

    def __init__(self, package_dir: Union[str, Path], device: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.package_dir = Path(package_dir)
        config_path = self.package_dir / "fusion_config.json"
        aml_path = self.package_dir / "aml_classifier.pkl"
        pubmed_dir = self.package_dir / "pubmedbert"

        if not config_path.is_file():
            raise FileNotFoundError(f"Missing fusion_config.json in {self.package_dir}")
        self.fusion_config: Dict[str, Any] = json.loads(config_path.read_text())
        self.alpha_star = float(self.fusion_config["alpha_star"])
        self.threshold_star = float(self.fusion_config["threshold_star"])
        self.max_length = int(
            self.fusion_config.get("models", {})
            .get("dl_stream", {})
            .get("max_length", dl_utils.DEFAULT_BIOCAUSE_MAX_LENGTH)
        )
        self.max_between_chars = int(
            self.fusion_config.get("post_processing", {})
            .get("linguistic_filter", {})
            .get("max_between_chars", 400)
        )

        self.aml_pipeline = load_aml_pipeline(aml_path)
        if not pubmed_dir.is_dir():
            raise FileNotFoundError(f"Missing PubMedBERT artifact directory: {pubmed_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(pubmed_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(pubmed_dir), local_files_only=True
        )
        if device is None:
            device = str(dl_utils.get_device())
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_marked_texts(
        self,
        df: pd.DataFrame,
        section_cache: Optional[Dict[str, str]] = None,
        raw_dir: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        if "marked_text" in df.columns and not df["marked_text"].isna().any():
            texts = df["marked_text"].astype(str).tolist()
            for idx, txt in enumerate(texts):
                for marker in dl_utils.ENTITY_MARKERS:
                    if marker not in txt:
                        raise ValueError(
                            f"Candidate {df.iloc[idx]['candidate_id']}: missing marker '{marker}' in marked_text."
                        )
            return texts

        req = [
            "section_id",
            "arg1_start",
            "arg1_end",
            "arg2_start",
            "arg2_end",
            "arg1_text",
            "arg2_text",
        ]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(
                "Input DataFrame must contain either 'marked_text' or source span columns "
                f"{req}; missing {missing}."
            )

        cache = {} if section_cache is None else section_cache
        resolved_raw_dir = (
            Path(raw_dir) if raw_dir is not None else (_ROOT_DIR / "data" / "raw")
        )
        texts = []
        for _, r in df.iterrows():
            sec_id = str(r["section_id"])
            sec_text = dl_utils.load_biocause_section_text(
                sec_id, raw_dir=str(resolved_raw_dir), cache=cache
            )
            marked = dl_utils.reconstruct_biocause_context_with_markers(
                section_text=sec_text,
                arg1_start=int(r["arg1_start"]),
                arg1_end=int(r["arg1_end"]),
                arg2_start=int(r["arg2_start"]),
                arg2_end=int(r["arg2_end"]),
                arg1_text=str(r["arg1_text"]),
                arg2_text=str(r["arg2_text"]),
                candidate_id=str(r["candidate_id"]),
                max_length=self.max_length,
                tokenizer=self.tokenizer,
            )
            texts.append(marked)
        return texts

    def predict_candidates(
        self,
        df_candidates: pd.DataFrame,
        batch_size: int = 16,
        section_cache: Optional[Dict[str, str]] = None,
        raw_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Run aligned AML + PubMedBERT + frozen hybrid scoring on BioCause candidates."""
        import torch

        df = _validate_aml_input_df(df_candidates).copy().reset_index(drop=True)
        if df["candidate_id"].isna().any() or df["candidate_id"].duplicated().any():
            raise ValueError("candidate_id must be non-null and unique for inference.")

        candidate_ids = df["candidate_id"].astype(str).tolist()
        marked_texts = self._prepare_marked_texts(
            df, section_cache=section_cache, raw_dir=raw_dir
        )

        # 1. AML Stream probabilities on identical candidate order
        aml_prob = np.asarray(self.aml_pipeline.predict_proba(df)[:, 1], dtype=np.float64)

        # 2. PubMedBERT Stream probabilities on identical candidate order
        dl_prob_chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(marked_texts), batch_size):
                batch_texts = marked_texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits.detach().cpu().numpy()
                dl_prob_chunks.append(logits_to_positive_probabilities(logits))
        dl_prob = np.concatenate(dl_prob_chunks, axis=0)

        if len(aml_prob) != len(dl_prob) or len(aml_prob) != len(candidate_ids):
            raise RuntimeError("Candidate alignment length mismatch across AML and DL streams.")
        if not (np.isfinite(aml_prob).all() and np.isfinite(dl_prob).all()):
            raise RuntimeError("Non-finite probability produced during inference.")

        # 3. Frozen linear probability fusion & threshold application
        hybrid_prob = dl_utils.fuse_probabilities(
            prob_aml=aml_prob, prob_dl=dl_prob, alpha=self.alpha_star
        )
        raw_hybrid_pred = (hybrid_prob >= self.threshold_star).astype(int)

        # 4. Linguistic sanity filter consistent with DL_04
        is_neg = (
            df["between_text"]
            .fillna("")
            .astype(str)
            .apply(lambda txt: bool(NEGATION_PATTERN.search(txt)))
        )
        is_dist = df["between_text"].fillna("").astype(str).str.len() > self.max_between_chars
        sanitized_pred = ((raw_hybrid_pred == 1) & (~is_neg) & (~is_dist)).astype(int)

        return pd.DataFrame(
            {
                "candidate_id": candidate_ids,
                "aml_prob": aml_prob,
                "dl_prob": dl_prob,
                "hybrid_prob": hybrid_prob,
                "hybrid_pred": raw_hybrid_pred,
                "flag_negation": is_neg.to_numpy(dtype=bool),
                "flag_distance": is_dist.to_numpy(dtype=bool),
                "sanitized_pred": sanitized_pred,
            }
        )

    def build_causal_dag(
        self,
        df_candidates: pd.DataFrame,
        df_predictions: pd.DataFrame,
        split_name: str = "inference",
        use_sanitized: bool = True,
    ) -> Tuple[pd.DataFrame, nx.DiGraph, List[Dict[str, Any]]]:
        """Convert positive predictions into 17-column records and an acyclic DAG."""
        pred_col = "sanitized_pred" if use_sanitized else "hybrid_pred"
        merged = df_candidates.merge(df_predictions, on="candidate_id", how="inner")
        if len(merged) != len(df_candidates):
            raise ValueError("Candidate ID mismatch between candidate table and predictions.")

        pos_df = merged[merged[pred_col] == 1]
        records: List[Dict[str, Any]] = []
        for _, r in pos_df.iterrows():
            records.append(
                dl_utils.build_unified_relation_record(
                    candidate_id=str(r["candidate_id"]),
                    article_id=str(r["article_id"]),
                    section_id=str(r["section_id"]),
                    cause_id=str(r["arg1_id"]),
                    cause_text=str(r["arg1_text"]),
                    cause_start=int(r["arg1_start"]),
                    cause_end=int(r["arg1_end"]),
                    effect_id=str(r["arg2_id"]),
                    effect_text=str(r["arg2_text"]),
                    effect_start=int(r["arg2_start"]),
                    effect_end=int(r["arg2_end"]),
                    confidence=float(r["hybrid_prob"]),
                    extraction_method="hybrid_probability_fusion",
                    evidence_text="" if pd.isna(r["between_text"]) else str(r["between_text"]),
                    split=str(r.get("split", split_name)),
                    direction="cause->effect",
                    provenance=(
                        f"PubMedBERT_alpha{self.alpha_star:.2f}"
                        f"+LogisticRegression_1to2_tau{self.threshold_star:.4f}"
                    ),
                )
            )
        df_records = pd.DataFrame(records, columns=dl_utils.UNIFIED_RELATION_SCHEMA)

        G_raw = nx.DiGraph()
        for _, r in df_records.iterrows():
            u = f"{r['section_id']}_{r['cause_id']}"
            v = f"{r['section_id']}_{r['effect_id']}"
            if not G_raw.has_node(u):
                G_raw.add_node(u, display_text=r["cause_text"], section_id=r["section_id"])
            if not G_raw.has_node(v):
                G_raw.add_node(v, display_text=r["effect_text"], section_id=r["section_id"])
            if G_raw.has_edge(u, v):
                G_raw[u][v]["evidence_count"] += 1
                G_raw[u][v]["confidence"] = max(
                    G_raw[u][v]["confidence"], float(r["confidence"])
                )
            else:
                G_raw.add_edge(
                    u,
                    v,
                    confidence=float(r["confidence"]),
                    evidence_count=1,
                    extraction_method=r["extraction_method"],
                )

        G_dag, audit_log = enforce_dag_acyclicity(G_raw, weight_attr="confidence")
        return df_records, G_dag, audit_log
