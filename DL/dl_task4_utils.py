"""Utility module for Phase 2 Task 4: Hybrid Probability Fusion and Graph Integration.

Provides tested, reusable helper logic for:
1. Prediction dataframe schema and contract validation.
2. Dual-stream candidate alignment and exact model-eligible membership verification.
3. Isolated split metadata streaming and label integrity verification.
4. Lexicographic validation hyperparameter selection with Macro-F1 and deterministic tie-breaking.
5. Linguistic sanity checks with auditable non-overlapping rejection diagnostics.
6. UnifiedRelationRecord formatting and schema conformance.
7. Prediction and relation export verification.
8. Directed acyclic graph (DAG) pruning diagnostics and DiscreteBayesianNetwork compilation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import precision_recall_fscore_support

import dl_utils
from causal_extractors import NEGATION_PATTERN


# ============================================================================
# 1. Prediction Dataframe & Stream Alignment Contracts
# ============================================================================

def validate_prediction_dataframe(
    df: pd.DataFrame,
    expected_rows: int,
    split_name: str,
    stream_name: str,
    prob_col: str,
    pred_col: str,
) -> None:
    """Validate prediction dataframe against strict Task 4 input contracts."""
    req_cols = ["candidate_id", "true_label", prob_col, pred_col, "split"]
    missing_cols = set(req_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"[{stream_name} {split_name}] Missing required columns: {sorted(list(missing_cols))}")
    if len(df) != expected_rows:
        raise ValueError(f"[{stream_name} {split_name}] Row count mismatch: expected {expected_rows}, got {len(df)}")
    if df["candidate_id"].isna().any() or (df["candidate_id"].astype(str).str.strip() == "").any():
        raise ValueError(f"[{stream_name} {split_name}] Found null or empty candidate_id values")
    if df["candidate_id"].duplicated().any():
        n_dup = int(df["candidate_id"].duplicated().sum())
        raise ValueError(f"[{stream_name} {split_name}] Found {n_dup} duplicate candidate_id values")
    if not (df["split"] == split_name).all():
        raise ValueError(f"[{stream_name} {split_name}] Non-matching split labels in split column")
    if not set(df["true_label"].unique()).issubset({0, 1}):
        raise ValueError(f"[{stream_name} {split_name}] Non-binary true_label values: {df['true_label'].unique()}")
    if not set(df[pred_col].unique()).issubset({0, 1}):
        raise ValueError(f"[{stream_name} {split_name}] Non-binary prediction values: {df[pred_col].unique()}")
    if not np.isfinite(df[prob_col]).all():
        raise ValueError(f"[{stream_name} {split_name}] Non-finite values in {prob_col}")
    if not df[prob_col].between(0.0, 1.0).all():
        raise ValueError(f"[{stream_name} {split_name}] Probabilities outside [0.0, 1.0] in {prob_col}")


def validate_dual_stream_alignment(
    df_aml: pd.DataFrame,
    df_dl: pd.DataFrame,
    split_name: str,
) -> None:
    """Assert candidate ID ordering and ground-truth label agreement across streams."""
    if len(df_aml) != len(df_dl):
        raise ValueError(f"[{split_name}] Length mismatch across streams: AML ({len(df_aml)}) vs DL ({len(df_dl)})")
    if not (df_aml["candidate_id"].astype(str).values == df_dl["candidate_id"].astype(str).values).all():
        raise ValueError(f"[{split_name}] Candidate ID order or identity mismatch across streams")
    if not (df_aml["true_label"].values == df_dl["true_label"].values).all():
        raise ValueError(f"[{split_name}] Ground truth label discrepancy across streams")


# ============================================================================
# 2. Authoritative Model-Eligible Membership & Metadata Isolation
# ============================================================================

def validate_and_load_authoritative_manifest(
    manifest_path: Union[str, Path],
    split_name: str,
    expected_total: int,
    expected_pos: Optional[int] = None,
    expected_neg: Optional[int] = None,
) -> pd.DataFrame:
    """Validate and load an authoritative Task 1 model-eligible candidate split manifest."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Authoritative Task 1 prepared {split_name} split file not found: {path}.\n"
            f"In strict adherence to requirements, expected candidate IDs must not be "
            f"inferred or fabricated."
        )

    df = pd.read_csv(path)
    if "candidate_id" not in df.columns:
        raise ValueError(f"Manifest {path} missing required 'candidate_id' column")

    # Reject null, empty, or duplicate candidate IDs
    if df["candidate_id"].isna().any():
        raise ValueError(f"Manifest {path} contains null candidate_id values")
    cids = df["candidate_id"].astype(str).str.strip()
    if (cids == "").any():
        raise ValueError(f"Manifest {path} contains empty candidate_id values")
    if cids.duplicated().any():
        n_dup = int(cids.duplicated().sum())
        raise ValueError(f"Manifest {path} contains {n_dup} duplicate candidate_id values")

    # Validate exact expected row count and unique ID count
    if len(df) != expected_total:
        raise ValueError(f"Manifest {path} row count mismatch: expected {expected_total}, got {len(df)}")
    if len(cids.unique()) != expected_total:
        raise ValueError(f"Manifest {path} unique candidate_id count mismatch: expected {expected_total}, got {len(cids.unique())}")

    # Validate split identity if manifest includes a split field
    if "split" in df.columns:
        valid_split_names = {split_name}
        if split_name in ("val", "validation"):
            valid_split_names = {"val", "validation"}
        elif split_name in ("test",):
            valid_split_names = {"test"}
        elif split_name in ("train", "training"):
            valid_split_names = {"train", "training"}

        splits_in_manifest = set(df["split"].astype(str).str.strip().unique())
        if not splits_in_manifest.issubset(valid_split_names):
            raise ValueError(f"Manifest {path} contains unexpected split values: {splits_in_manifest} (expected {valid_split_names})")

    # Validate binary labels and expected positive/negative counts if label column exists
    label_col = None
    for cand_col in ["label", "true_label"]:
        if cand_col in df.columns:
            label_col = cand_col
            break

    if label_col is not None:
        if not set(df[label_col].unique()).issubset({0, 1}):
            raise ValueError(f"Manifest {path} contains non-binary values in '{label_col}': {set(df[label_col].unique())}")
        if expected_pos is not None:
            actual_pos = int((df[label_col] == 1).sum())
            if actual_pos != expected_pos:
                raise ValueError(f"Manifest {path} positive count mismatch: expected {expected_pos}, got {actual_pos}")
        if expected_neg is not None:
            actual_neg = int((df[label_col] == 0).sum())
            if actual_neg != expected_neg:
                raise ValueError(f"Manifest {path} negative count mismatch: expected {expected_neg}, got {actual_neg}")

    return df


def load_authoritative_eligible_candidate_ids(
    split_manifest_path: Union[str, Path],
    split_name: str,
    expected_count: int,
    expected_pos: Optional[int] = None,
    expected_neg: Optional[int] = None,
) -> Set[str]:
    """Load authoritative model-eligible candidate IDs from Task 1 prepared split file."""
    df = validate_and_load_authoritative_manifest(
        manifest_path=split_manifest_path,
        split_name=split_name,
        expected_total=expected_count,
        expected_pos=expected_pos,
        expected_neg=expected_neg,
    )
    return set(df["candidate_id"].astype(str).values)


def validate_exact_eligible_membership(
    pred_candidate_ids: Sequence[Any],
    expected_eligible_candidate_ids: Set[str],
    split_name: str,
    stream_name: str,
) -> None:
    """Assert exact set equality between prediction candidate IDs and authoritative model-eligible IDs."""
    pred_set = set(str(cid) for cid in pred_candidate_ids)
    if pred_set != expected_eligible_candidate_ids:
        missing = expected_eligible_candidate_ids - pred_set
        extra = pred_set - expected_eligible_candidate_ids
        err_msg = [f"[{stream_name} {split_name}] Exact candidate membership mismatch!"]
        if missing:
            err_msg.append(f"  Missing {len(missing)} eligible IDs (sample: {list(missing)[:3]})")
        if extra:
            err_msg.append(f"  Found {len(extra)} ineligible/extra IDs (sample: {list(extra)[:3]})")
        raise ValueError("\n".join(err_msg))


def load_split_candidate_metadata(
    metadata_path: Union[str, Path],
    split_name: str,
    usecols: Optional[Sequence[str]] = None,
    chunksize: int = 1000,
) -> pd.DataFrame:
    """Stream and filter candidate metadata, loading and retaining only the requested split records.
    
    Guarantees that non-matching split records, metadata, and labels are never retained in memory.
    If usecols is provided, only those specific columns are loaded.
    """
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate metadata file not found: {path}")

    load_cols = None
    drop_split_after = False
    if usecols is not None:
        load_cols = list(usecols)
        if "split" not in load_cols:
            load_cols.append("split")
            drop_split_after = True

    chunks = []
    for chunk in pd.read_csv(path, usecols=load_cols, keep_default_na=False, chunksize=chunksize):
        if "split" not in chunk.columns:
            raise ValueError(f"Metadata file {path} missing required 'split' column")
        matching = chunk[chunk["split"] == split_name]
        if not matching.empty:
            if drop_split_after:
                matching = matching.drop(columns=["split"])
            chunks.append(matching)

    if not chunks:
        raise ValueError(f"No candidates found for split '{split_name}' in {path}")

    df_split = pd.concat(chunks, ignore_index=True)
    if not drop_split_after and not (df_split["split"] == split_name).all():
        raise ValueError(f"Internal error: Leaked non-{split_name} split records during metadata loading!")
    return df_split


def validate_metadata_split_and_label_integrity(
    df_pred: pd.DataFrame,
    df_meta_split: pd.DataFrame,
    split_name: str,
) -> None:
    """Validate prediction candidate labels against isolated split metadata."""
    if "candidate_id" not in df_meta_split.columns:
        raise ValueError("Metadata dataframe must contain 'candidate_id' column")
    if df_meta_split["candidate_id"].isna().any() or (df_meta_split["candidate_id"].astype(str).str.strip() == "").any():
        raise ValueError("Metadata contains null or empty candidate_id values")
    if df_meta_split["candidate_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate candidate_id values")
    if "split" in df_meta_split.columns and not (df_meta_split["split"] == split_name).all():
        raise ValueError(f"Metadata contains non-{split_name} split rows!")

    meta_indexed = df_meta_split.set_index("candidate_id")
    pred_ids = df_pred["candidate_id"].values

    missing_ids = set(pred_ids) - set(meta_indexed.index)
    if missing_ids:
        raise ValueError(f"[{split_name}] {len(missing_ids)} prediction candidate IDs not found in {split_name} metadata")

    label_col = "label" if "label" in meta_indexed.columns else "true_label"
    meta_labels = meta_indexed.loc[pred_ids, label_col].values
    if not (df_pred["true_label"].values == meta_labels).all():
        mismatched_labels = int(np.sum(df_pred["true_label"].values != meta_labels))
        raise ValueError(f"[{split_name}] {mismatched_labels} candidate labels disagree with metadata")


# ============================================================================
# 3. Lexicographic Validation Selection with Tie-Breaking
# ============================================================================

def is_better_configuration(
    cand_f1_pos: float,
    cand_macro_f1: float,
    cand_alpha: float,
    cand_tau: float,
    best_f1_pos: float,
    best_macro_f1: float,
    best_alpha: float,
    best_tau: float,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """Determine whether candidate configuration is strictly better under the lexicographic selection rule.
    
    Rule:
    1. Higher positive-class F1.
    2. If positive F1 equal within tolerance (np.isclose), higher Macro-F1.
    3. If both metrics equal within tolerance, deterministic tie-break: smaller alpha, then smaller threshold tau.
    """
    if best_f1_pos < 0.0:
        return True

    # 1. Primary objective: Positive-class F1
    if not np.isclose(cand_f1_pos, best_f1_pos, rtol=rtol, atol=atol):
        return cand_f1_pos > best_f1_pos

    # 2. Secondary objective: Macro-F1
    if not np.isclose(cand_macro_f1, best_macro_f1, rtol=rtol, atol=atol):
        return cand_macro_f1 > best_macro_f1

    # 3. Deterministic tie-break: Smaller alpha
    if not np.isclose(cand_alpha, best_alpha, rtol=rtol, atol=atol):
        return cand_alpha < best_alpha

    # 4. Deterministic tie-break: Smaller tau
    if not np.isclose(cand_tau, best_tau, rtol=rtol, atol=atol):
        return cand_tau < best_tau

    return False


def select_validation_fusion_hyperparameters(
    val_y_true: np.ndarray,
    val_p_aml: np.ndarray,
    val_p_dl: np.ndarray,
    alpha_grid: Optional[Sequence[float]] = None,
    tau_grid: Optional[Sequence[float]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[float, float, float, float, List[Dict[str, float]]]:
    """Execute validation grid search using the documented lexicographic selection rule."""
    if alpha_grid is None:
        alpha_grid = np.round(np.arange(0.0, 1.05, 0.05), 2)
    if tau_grid is None:
        tau_grid = np.round(np.arange(0.10, 0.95, 0.05), 2)

    best_f1_pos = -1.0
    best_macro_f1 = -1.0
    best_alpha = 0.50
    best_tau = 0.50
    search_records: List[Dict[str, float]] = []

    for alpha in alpha_grid:
        p_hyb = dl_utils.fuse_probabilities(prob_aml=val_p_aml, prob_dl=val_p_dl, alpha=float(alpha))
        for tau in tau_grid:
            y_pred = (p_hyb >= float(tau)).astype(int)
            prec, rec, f1_pos, _ = precision_recall_fscore_support(val_y_true, y_pred, average="binary", zero_division=0)
            _, _, macro_f1, _ = precision_recall_fscore_support(val_y_true, y_pred, average="macro", zero_division=0)

            record = {
                "alpha": float(alpha),
                "tau": float(tau),
                "pos_precision": float(prec),
                "pos_recall": float(rec),
                "pos_f1": float(f1_pos),
                "macro_f1": float(macro_f1),
            }
            search_records.append(record)

            if is_better_configuration(
                cand_f1_pos=float(f1_pos),
                cand_macro_f1=float(macro_f1),
                cand_alpha=float(alpha),
                cand_tau=float(tau),
                best_f1_pos=best_f1_pos,
                best_macro_f1=best_macro_f1,
                best_alpha=best_alpha,
                best_tau=best_tau,
                rtol=rtol,
                atol=atol,
            ):
                best_f1_pos = float(f1_pos)
                best_macro_f1 = float(macro_f1)
                best_alpha = float(alpha)
                best_tau = float(tau)

    return best_alpha, best_tau, best_f1_pos, best_macro_f1, search_records


# ============================================================================
# 4. Linguistic Filtering & Non-Overlapping Diagnostics
# ============================================================================

def apply_linguistic_filters(
    df: pd.DataFrame,
    max_between_chars: int = 400,
    pred_col: str = "raw_pred",
) -> pd.DataFrame:
    """Apply negation and distance filters with auditable, non-overlapping diagnostics.
    
    Raises ValueError if between_text is null or NaN to prevent silent pass-through.
    """
    if "between_text" not in df.columns:
        raise ValueError("Missing 'between_text' column required for linguistic sanity checks")

    df_out = df.copy()
    neg_flags = []
    dist_flags = []
    reasons = []

    for idx, r in df_out.iterrows():
        raw_val = r["between_text"]
        if pd.isna(raw_val) or raw_val is None:
            raise ValueError(
                f"Missing between_text encountered for candidate {r.get('candidate_id', idx)}. "
                "Missing metadata must not be silently coerced to string."
            )
        b_txt = str(raw_val)
        if b_txt.strip() == "" or b_txt.strip().lower() == "nan":
            raise ValueError(
                f"Invalid or empty between_text ('{raw_val}') encountered for candidate {r.get('candidate_id', idx)}. "
                "Missing metadata must not be silently coerced to string."
            )
        is_neg = bool(NEGATION_PATTERN.search(b_txt))
        is_dist = len(b_txt) > max_between_chars

        neg_flags.append(is_neg)
        dist_flags.append(is_dist)

        r_list = []
        if is_neg:
            r_list.append("negation_detected")
        if is_dist:
            r_list.append("distance_window_exceeded")
        reasons.append(";".join(r_list) if r_list else "none")

    df_out["flag_negation"] = neg_flags
    df_out["flag_distance"] = dist_flags
    df_out["pruned_reason"] = reasons
    df_out["sanitized_pred"] = (
        (df_out[pred_col] == 1) & (~df_out["flag_negation"]) & (~df_out["flag_distance"])
    ).astype(int)

    return df_out


def compute_linguistic_diagnostics(
    df: pd.DataFrame,
    pred_col: str = "raw_hybrid_pred",
    sanitized_col: str = "sanitized_pred",
) -> Dict[str, int]:
    """Compute exact, non-overlapping rejection diagnostics for positive predictions."""
    is_initial_pos = (df[pred_col] == 1)
    n_initial = int(is_initial_pos.sum())

    pos_neg_only = is_initial_pos & df["flag_negation"] & (~df["flag_distance"])
    pos_dist_only = is_initial_pos & (~df["flag_negation"]) & df["flag_distance"]
    pos_both = is_initial_pos & df["flag_negation"] & df["flag_distance"]

    n_neg_only = int(pos_neg_only.sum())
    n_dist_only = int(pos_dist_only.sum())
    n_both = int(pos_both.sum())
    n_total_rejected = n_neg_only + n_dist_only + n_both
    n_surviving = int((df[sanitized_col] == 1).sum())

    if n_initial != n_surviving + n_total_rejected:
        raise ValueError(
            f"Linguistic diagnostic mismatch: {n_initial} initial != {n_surviving} surviving + {n_total_rejected} rejected"
        )

    return {
        "initial_positive": n_initial,
        "rejected_negation_only": n_neg_only,
        "rejected_distance_only": n_dist_only,
        "rejected_both": n_both,
        "total_unique_rejected": n_total_rejected,
        "surviving_positive": n_surviving,
    }


# ============================================================================
# 5. Export Verification Functions
# ============================================================================

def verify_prediction_export(
    filepath: Union[str, Path],
    expected_rows: int,
    split_name: str,
    source_candidate_ids: Sequence[Any],
    source_true_labels: Sequence[int],
    source_aml_probs: Sequence[float],
    source_dl_probs: Sequence[float],
    frozen_alpha: float,
    frozen_tau: float,
    source_aml_preds: Optional[Sequence[int]] = None,
    source_dl_preds: Optional[Sequence[int]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    include_sanitized_col: bool = False,
) -> pd.DataFrame:
    """Read back and verify exported hybrid prediction CSV against source arrays and frozen config."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Exported prediction file not found: {path}")

    df_exp = pd.read_csv(path)

    # 1. Row count
    if len(df_exp) != expected_rows:
        raise ValueError(f"[{split_name} Export] Row count mismatch: expected {expected_rows}, got {len(df_exp)}")

    # 2. Required columns and schema
    base_cols = ["candidate_id", "true_label", "aml_prob", "dl_prob", "hybrid_prob", "aml_pred", "dl_pred", "hybrid_pred", "split"]
    if include_sanitized_col:
        base_cols.append("sanitized_pred")
    missing_cols = set(base_cols) - set(df_exp.columns)
    if missing_cols:
        raise ValueError(f"[{split_name} Export] Missing required columns: {sorted(list(missing_cols))}")

    # 3. No nulls or empty values in required fields
    for col in base_cols:
        if df_exp[col].isna().any():
            raise ValueError(f"[{split_name} Export] Found null values in column '{col}'")
        if df_exp[col].dtype == object and (df_exp[col].astype(str).str.strip() == "").any():
            raise ValueError(f"[{split_name} Export] Found empty string values in column '{col}'")

    # 4. Candidate IDs: no duplicates, non-empty, and match source exactly in same order
    if df_exp["candidate_id"].duplicated().any():
        n_dup = int(df_exp["candidate_id"].duplicated().sum())
        raise ValueError(f"[{split_name} Export] Found {n_dup} duplicate candidate_id values")
    if not (df_exp["candidate_id"].astype(str).values == np.array(source_candidate_ids).astype(str)).all():
        raise ValueError(f"[{split_name} Export] Candidate ID ordering or content mismatch with source")

    # 5. True labels match source and are binary
    if not (df_exp["true_label"].values == np.array(source_true_labels)).all():
        raise ValueError(f"[{split_name} Export] True labels do not match source labels")
    if not set(df_exp["true_label"].unique()).issubset({0, 1}):
        raise ValueError(f"[{split_name} Export] Non-binary values in true_label")

    # 6. Probabilities finite and within [0, 1]
    for p_col in ["aml_prob", "dl_prob", "hybrid_prob"]:
        if not np.isfinite(df_exp[p_col]).all():
            raise ValueError(f"[{split_name} Export] Non-finite values in {p_col}")
        if not df_exp[p_col].between(0.0, 1.0).all():
            raise ValueError(f"[{split_name} Export] Values outside [0.0, 1.0] in {p_col}")

    # 7. Compare exported AML and DL probabilities directly against source arrays
    src_aml_p = np.array(source_aml_probs, dtype=float)
    src_dl_p = np.array(source_dl_probs, dtype=float)
    if not np.allclose(df_exp["aml_prob"].values, src_aml_p, rtol=rtol, atol=atol):
        max_diff = float(np.max(np.abs(df_exp["aml_prob"].values - src_aml_p)))
        raise ValueError(f"[{split_name} Export] Exported aml_prob does not match source AML probabilities! Max diff: {max_diff}")
    if not np.allclose(df_exp["dl_prob"].values, src_dl_p, rtol=rtol, atol=atol):
        max_diff = float(np.max(np.abs(df_exp["dl_prob"].values - src_dl_p)))
        raise ValueError(f"[{split_name} Export] Exported dl_prob does not match source DL probabilities! Max diff: {max_diff}")

    # 8. Independent recomputation of fusion formula from SOURCE arrays
    recomputed_from_source = frozen_alpha * src_dl_p + (1.0 - frozen_alpha) * src_aml_p
    if not np.allclose(df_exp["hybrid_prob"].values, recomputed_from_source, rtol=rtol, atol=atol):
        max_diff = float(np.max(np.abs(df_exp["hybrid_prob"].values - recomputed_from_source)))
        raise ValueError(f"[{split_name} Export] Exported hybrid_prob mismatch with source-recomputed formula! Max diff: {max_diff}")

    # 9. Predictions are binary and match
    for pred_col in ["aml_pred", "dl_pred", "hybrid_pred"]:
        if not set(df_exp[pred_col].unique()).issubset({0, 1}):
            raise ValueError(f"[{split_name} Export] Non-binary values in {pred_col}")

    if source_aml_preds is not None:
        if not (df_exp["aml_pred"].values == np.array(source_aml_preds, dtype=int)).all():
            raise ValueError(f"[{split_name} Export] Exported aml_pred does not match source aml_pred array")

    if source_dl_preds is not None:
        if not (df_exp["dl_pred"].values == np.array(source_dl_preds, dtype=int)).all():
            raise ValueError(f"[{split_name} Export] Exported dl_pred does not match source dl_pred array")

    expected_hybrid_pred = (df_exp["hybrid_prob"].values >= frozen_tau).astype(int)
    if not (df_exp["hybrid_pred"].values == expected_hybrid_pred).all():
        mismatched = int(np.sum(df_exp["hybrid_pred"].values != expected_hybrid_pred))
        raise ValueError(f"[{split_name} Export] hybrid_pred does not match (hybrid_prob >= frozen_tau) for {mismatched} rows")

    if include_sanitized_col:
        if not set(df_exp["sanitized_pred"].unique()).issubset({0, 1}):
            raise ValueError(f"[{split_name} Export] Non-binary values in sanitized_pred")

    # 10. Split column
    if not (df_exp["split"] == split_name).all():
        raise ValueError(f"[{split_name} Export] Split column contains values other than '{split_name}'")

    return df_exp


def verify_relation_export(
    filepath: Union[str, Path],
    expected_count: int,
    sanitized_candidate_ids: Sequence[Any],
    traceability_map: Optional[Dict[str, str]] = None,
    candidate_id_to_relation_id: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Read back and verify the 17-column UnifiedRelationRecord export with independent candidate traceability."""
    if traceability_map is None and candidate_id_to_relation_id is not None:
        traceability_map = candidate_id_to_relation_id
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Exported relation file not found: {path}")

    df_rels = pd.read_csv(path)

    # 1. Exact schema and column order against UNIFIED_RELATION_SCHEMA
    if list(df_rels.columns) != dl_utils.UNIFIED_RELATION_SCHEMA:
        raise ValueError(
            f"Relation schema mismatch!\nExpected: {dl_utils.UNIFIED_RELATION_SCHEMA}\nGot: {list(df_rels.columns)}"
        )

    # 2. Relation count matches expected count and sanitized positive candidates count
    if len(df_rels) != expected_count:
        raise ValueError(f"Relation export row count mismatch: expected {expected_count}, got {len(df_rels)}")
    if len(df_rels) != len(sanitized_candidate_ids):
        raise ValueError(
            f"Relation export count ({len(df_rels)}) does not match sanitized positive candidates count ({len(sanitized_candidate_ids)})"
        )

    # 3. Required identifier fields non-null and non-empty
    id_fields = ["relation_id", "article_id", "section_id", "cause_id", "effect_id"]
    for f in id_fields:
        if df_rels[f].isna().any() or (df_rels[f].astype(str).str.strip() == "").any():
            raise ValueError(f"Found null or empty values in required identifier field '{f}'")

    # 4. relation_id unique
    if df_rels["relation_id"].duplicated().any():
        n_dup = int(df_rels["relation_id"].duplicated().sum())
        raise ValueError(f"Found {n_dup} duplicate relation_id values in relation export")

    # 5. Cause and effect IDs are not identical (reject self-loops)
    same_entity = df_rels["cause_id"] == df_rels["effect_id"]
    if same_entity.any():
        n_same = int(same_entity.sum())
        raise ValueError(f"Found {n_same} self-referential relations (cause_id == effect_id)")

    # 6. Direction exactly cause->effect
    if not (df_rels["direction"] == "cause->effect").all():
        raise ValueError("Found relation records with direction other than 'cause->effect'")

    # 7. Split exactly test
    if not (df_rels["split"] == "test").all():
        raise ValueError("Found relation records with split other than 'test'")

    # 8. Confidence: numeric, finite, within [0.0, 1.0]
    conf = pd.to_numeric(df_rels["confidence"], errors="coerce")
    if conf.isna().any() or not np.isfinite(conf).all() or not conf.between(0.0, 1.0).all():
        raise ValueError("Confidence values not finite or outside [0.0, 1.0]")

    # 9. Offsets: numeric, finite, non-null, integer-valued, non-negative, and start < end
    offset_cols = ["cause_start", "cause_end", "effect_start", "effect_end"]
    for col in offset_cols:
        val_numeric = pd.to_numeric(df_rels[col], errors="coerce")
        if val_numeric.isna().any() or not np.isfinite(val_numeric).all():
            raise ValueError(f"Non-numeric, infinite, or null values in offset column '{col}'")
        if (val_numeric < 0).any():
            raise ValueError(f"Negative offsets found in column '{col}'")
        if not val_numeric.apply(lambda x: float(x).is_integer()).all():
            raise ValueError(f"Fractional offsets found in column '{col}'")

    if (df_rels["cause_start"] >= df_rels["cause_end"]).any():
        raise ValueError("Invalid cause span order: cause_start >= cause_end")
    if (df_rels["effect_start"] >= df_rels["effect_end"]).any():
        raise ValueError("Invalid effect span order: effect_start >= effect_end")

    # 10. Independent candidate traceability
    expected_rel_ids = {dl_utils.generate_relation_id(str(cid)) for cid in sanitized_candidate_ids}
    if len(expected_rel_ids) != len(sanitized_candidate_ids):
        raise ValueError("Collision detected in independent relation ID generation from candidate IDs!")

    actual_rel_ids = set(df_rels["relation_id"].astype(str))
    if actual_rel_ids != expected_rel_ids:
        missing = expected_rel_ids - actual_rel_ids
        extra = actual_rel_ids - expected_rel_ids
        raise ValueError(
            f"Relation export candidate traceability mismatch!\n"
            f"Missing relation IDs: {len(missing)}, Extra relation IDs: {len(extra)}"
        )

    if traceability_map is not None:
        map_rel_ids = set(traceability_map.values())
        if actual_rel_ids != map_rel_ids:
            raise ValueError("Exported relation IDs do not match provided traceability map!")

    return df_rels


# ============================================================================
# 6. Graph Compilation & Diagnostics Verification
# ============================================================================

def compile_discrete_bayesian_network(G_dag: nx.DiGraph) -> DiscreteBayesianNetwork:
    """Compile G_dag into a DiscreteBayesianNetwork preserving all isolated and connected nodes."""
    bn = DiscreteBayesianNetwork()
    bn.add_nodes_from(G_dag.nodes())
    bn.add_edges_from(G_dag.edges())

    # Assert exact node-set and edge-set equality
    assert set(bn.nodes()) == set(G_dag.nodes()), "Node set mismatch in Bayesian network compilation!"
    assert set(bn.edges()) == set(G_dag.edges()), "Edge set mismatch in Bayesian network compilation!"
    assert nx.is_directed_acyclic_graph(bn), "Compiled Bayesian network is not a directed acyclic graph!"
    return bn


def verify_dag_and_pruning_diagnostics(
    G_pred: nx.DiGraph,
    G_dag: nx.DiGraph,
    audit_log: List[Dict[str, Any]],
    diag_filepath: Union[str, Path],
    dag_edges_filepath: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Verify graph properties and reconcile persisted diagnostics JSON."""
    assert nx.is_directed_acyclic_graph(G_dag), "Final graph G_dag is not a DAG!"

    n_initial_edges = G_pred.number_of_edges()
    n_final_edges = G_dag.number_of_edges()
    n_pruned = len(audit_log)

    assert n_initial_edges == n_final_edges + n_pruned, (
        f"Graph edge arithmetic mismatch: {n_initial_edges} initial != {n_final_edges} final + {n_pruned} pruned"
    )

    path = Path(diag_filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Diagnostics file not found: {path}")

    with open(path) as f:
        diag = json.load(f)

    assert diag["initial_candidate_edges"] == n_initial_edges, "Initial edge count mismatch in diagnostics"
    assert diag["final_dag_edges"] == n_final_edges, "Final DAG edge count mismatch in diagnostics"
    assert diag["pruned_edges_count"] == n_pruned, "Pruned edge count mismatch in diagnostics"

    # Reconcile pruning breakdown with audit log
    n_self_loops = sum(1 for e in audit_log if e.get("reason") == "self_loop")
    n_bidirectional = sum(1 for e in audit_log if "bidirectional_conflict" in e.get("reason", ""))
    n_cycle_feedback = sum(1 for e in audit_log if "cycle" in e.get("reason", ""))

    if "pruning_breakdown" in diag:
        assert diag["pruning_breakdown"].get("self_loops", 0) == n_self_loops, "Self-loop count mismatch in diagnostics"
        assert diag["pruning_breakdown"].get("bidirectional_conflicts", 0) == n_bidirectional, "Bidirectional count mismatch in diagnostics"
        assert diag["pruning_breakdown"].get("directed_cycle_feedback", 0) == n_cycle_feedback, "Cycle feedback count mismatch in diagnostics"

    for stage in ["before_pruning", "after_pruning"]:
        metrics = diag["edge_evaluation"][stage]
        for m in ["Precision", "Recall", "F1"]:
            val = metrics[m]
            assert np.isfinite(val) and 0.0 <= val <= 1.0, f"Invalid {m} in edge evaluation: {val}"

    # If dag_edges_filepath is provided, verify against G_dag edges
    if dag_edges_filepath is not None:
        p_edges = Path(dag_edges_filepath)
        if not p_edges.is_file():
            raise FileNotFoundError(f"Exported DAG edges file not found: {p_edges}")
        df_edges = pd.read_csv(p_edges)
        edge_pairs = set(zip(df_edges["source"].astype(str), df_edges["target"].astype(str)))
        expected_pairs = set((str(u), str(v)) for u, v in G_dag.edges())
        assert edge_pairs == expected_pairs, "Exported DAG edges CSV does not match NetworkX G_dag edges!"

    return diag
