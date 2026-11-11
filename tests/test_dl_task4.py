"""Focused unit tests for Phase 2 Task 4: Hybrid Probability Fusion and Graph Integration.

Verifies:
1. Prediction-file contract validation (schema, counts, nulls, duplicates, probabilities, binary labels).
2. Dual-stream candidate ID ordering and ground-truth label alignment.
3. Metadata split integrity and candidate population tracking.
4. Probability fusion at boundary and intermediate alpha values.
5. Linguistic sanity filtering with non-overlapping rejection diagnostics (negation, distance, both)
   and fail-fast missing metadata policy.
6. Standardized 17-column UnifiedRelationRecord schema conformance.
7. NetworkX DAG construction, self-loop pruning, bidirectional conflict resolution, and cycle breaking.
8. Bayesian network compilation and observational vs. interventional queries.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

# Ensure DL and AML modules are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
for p in [ROOT_DIR, ROOT_DIR / "AML", ROOT_DIR / "DL"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dl_utils
from causal_extractors import enforce_dag_acyclicity, NEGATION_PATTERN
import pgmpy
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# ============================================================================
# Reusable Task 4 Helper Functions Under Test
# ============================================================================

def validate_prediction_dataframe(
    df: pd.DataFrame,
    expected_rows: int,
    split_name: str,
    stream_name: str,
    prob_col: str,
    pred_col: str,
) -> None:
    """Validate prediction dataframe against strict Task 4 contracts."""
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
    if not (df_aml["candidate_id"].values == df_dl["candidate_id"].values).all():
        raise ValueError(f"[{split_name}] Candidate ID order or identity mismatch across streams")
    if not (df_aml["true_label"].values == df_dl["true_label"].values).all():
        raise ValueError(f"[{split_name}] Ground truth label discrepancy across streams")


def validate_metadata_split_integrity(
    df_pred: pd.DataFrame,
    df_meta: pd.DataFrame,
    split_name: str,
) -> None:
    """Validate prediction candidates against biocause_candidates.csv metadata."""
    if "candidate_id" not in df_meta.columns:
        raise ValueError("Metadata dataframe must contain 'candidate_id' column")
    if df_meta["candidate_id"].isna().any() or (df_meta["candidate_id"].astype(str).str.strip() == "").any():
        raise ValueError("Metadata contains null or empty candidate_id values")
    if df_meta["candidate_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate candidate_id values")

    meta_indexed = df_meta.set_index("candidate_id")
    pred_ids = df_pred["candidate_id"].values

    # Check for missing candidates
    missing_ids = set(pred_ids) - set(meta_indexed.index)
    if missing_ids:
        raise ValueError(f"[{split_name}] {len(missing_ids)} prediction candidate IDs not found in metadata")

    # Check split membership
    meta_splits = meta_indexed.loc[pred_ids, "split"].values
    if not (meta_splits == split_name).all():
        mismatched = np.sum(meta_splits != split_name)
        raise ValueError(f"[{split_name}] {mismatched} candidate IDs belong to wrong split in metadata")

    # Check label agreement
    meta_labels = meta_indexed.loc[pred_ids, "label"].values
    if not (df_pred["true_label"].values == meta_labels).all():
        mismatched_labels = np.sum(df_pred["true_label"].values != meta_labels)
        raise ValueError(f"[{split_name}] {mismatched_labels} candidate labels disagree with metadata")


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
                "Missing data must not be silently coerced to string."
            )
        b_txt = str(raw_val)
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


# ============================================================================
# Unit Test Cases
# ============================================================================

class TestTask4InputValidationAndAlignment(unittest.TestCase):
    """Test strict candidate alignment and input contract assertions."""

    def setUp(self):
        self.n_val = 10
        self.val_cands = [f"cand_val_{i:02d}" for i in range(self.n_val)]
        self.val_labels = [0, 1] * (self.n_val // 2)

        self.df_aml_valid = pd.DataFrame({
            "candidate_id": self.val_cands,
            "true_label": self.val_labels,
            "aml_prob": np.linspace(0.1, 0.9, self.n_val),
            "aml_pred": [0, 1] * (self.n_val // 2),
            "split": ["validation"] * self.n_val,
        })

        self.df_dl_valid = pd.DataFrame({
            "candidate_id": self.val_cands,
            "true_label": self.val_labels,
            "dl_prob": np.linspace(0.2, 0.8, self.n_val),
            "dl_pred": [0, 1] * (self.n_val // 2),
            "split": ["validation"] * self.n_val,
        })

        self.df_meta_valid = pd.DataFrame({
            "candidate_id": self.val_cands,
            "label": self.val_labels,
            "split": ["validation"] * self.n_val,
            "between_text": ["activates"] * self.n_val,
        })

    def test_valid_contracts_pass(self):
        """Correct schema and values must pass all validation checks without error."""
        validate_prediction_dataframe(self.df_aml_valid, self.n_val, "validation", "AML", "aml_prob", "aml_pred")
        validate_prediction_dataframe(self.df_dl_valid, self.n_val, "validation", "DL", "dl_prob", "dl_pred")
        validate_dual_stream_alignment(self.df_aml_valid, self.df_dl_valid, "validation")
        validate_metadata_split_integrity(self.df_aml_valid, self.df_meta_valid, "validation")

    def test_missing_column_raises_value_error(self):
        """Missing required column must raise ValueError."""
        df_bad = self.df_aml_valid.drop(columns=["aml_prob"])
        with self.assertRaises(ValueError) as ctx:
            validate_prediction_dataframe(df_bad, self.n_val, "validation", "AML", "aml_prob", "aml_pred")
        self.assertIn("Missing required columns", str(ctx.exception))

    def test_row_count_mismatch_raises_value_error(self):
        """Row count mismatch must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_prediction_dataframe(self.df_aml_valid, self.n_val + 5, "validation", "AML", "aml_prob", "aml_pred")
        self.assertIn("Row count mismatch", str(ctx.exception))

    def test_null_or_empty_candidate_id_raises_value_error(self):
        """Null or empty candidate_id must be rejected."""
        df_bad = self.df_aml_valid.copy()
        df_bad.loc[0, "candidate_id"] = ""
        with self.assertRaises(ValueError) as ctx:
            validate_prediction_dataframe(df_bad, self.n_val, "validation", "AML", "aml_prob", "aml_pred")
        self.assertIn("null or empty candidate_id", str(ctx.exception))

    def test_duplicate_candidate_id_raises_value_error(self):
        """Duplicate candidate IDs must be rejected."""
        df_bad = self.df_aml_valid.copy()
        df_bad.loc[1, "candidate_id"] = df_bad.loc[0, "candidate_id"]
        with self.assertRaises(ValueError) as ctx:
            validate_prediction_dataframe(df_bad, self.n_val, "validation", "AML", "aml_prob", "aml_pred")
        self.assertIn("duplicate candidate_id", str(ctx.exception))

    def test_non_binary_label_raises_value_error(self):
        """Non-binary label must be rejected."""
        df_bad = self.df_aml_valid.copy()
        df_bad.loc[0, "true_label"] = 2
        with self.assertRaises(ValueError) as ctx:
            validate_prediction_dataframe(df_bad, self.n_val, "validation", "AML", "aml_prob", "aml_pred")
        self.assertIn("Non-binary true_label", str(ctx.exception))

    def test_out_of_bounds_probability_raises_value_error(self):
        """Probabilities < 0 or > 1 must be rejected."""
        df_bad = self.df_aml_valid.copy()
        df_bad.loc[0, "aml_prob"] = 1.05
        with self.assertRaises(ValueError) as ctx:
            validate_prediction_dataframe(df_bad, self.n_val, "validation", "AML", "aml_prob", "aml_pred")
        self.assertIn("Probabilities outside [0.0, 1.0]", str(ctx.exception))

    def test_stream_alignment_detects_candidate_order_mismatch(self):
        """Stream alignment must detect candidate order discrepancies."""
        df_dl_bad = self.df_dl_valid.copy().iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError) as ctx:
            validate_dual_stream_alignment(self.df_aml_valid, df_dl_bad, "validation")
        self.assertIn("Candidate ID order or identity mismatch", str(ctx.exception))

    def test_stream_alignment_detects_label_discrepancy(self):
        """Stream alignment must detect ground-truth label disagreement."""
        df_dl_bad = self.df_dl_valid.copy()
        df_dl_bad.loc[0, "true_label"] = 1 - df_dl_bad.loc[0, "true_label"]
        with self.assertRaises(ValueError) as ctx:
            validate_dual_stream_alignment(self.df_aml_valid, df_dl_bad, "validation")
        self.assertIn("Ground truth label discrepancy", str(ctx.exception))

    def test_metadata_integrity_detects_wrong_split_membership(self):
        """Candidates belonging to a different split in metadata must be rejected."""
        df_meta_bad = self.df_meta_valid.copy()
        df_meta_bad.loc[0, "split"] = "train"
        with self.assertRaises(ValueError) as ctx:
            validate_metadata_split_integrity(self.df_aml_valid, df_meta_bad, "validation")
        self.assertIn("belong to wrong split", str(ctx.exception))

    def test_metadata_integrity_detects_missing_candidate(self):
        """Prediction candidates absent from metadata must be detected."""
        df_meta_incomplete = self.df_meta_valid.iloc[1:].copy()
        with self.assertRaises(ValueError) as ctx:
            validate_metadata_split_integrity(self.df_aml_valid, df_meta_incomplete, "validation")
        self.assertIn("not found in metadata", str(ctx.exception))


class TestTask4ProbabilityFusion(unittest.TestCase):
    """Test hybrid probability fusion formula and boundary conditions."""

    def test_pure_aml_fusion_alpha_zero(self):
        """Alpha = 0.0 must exactly equal AML probabilities."""
        aml_probs = np.array([0.1, 0.4, 0.7, 0.9])
        dl_probs = np.array([0.3, 0.5, 0.2, 0.8])
        fused = dl_utils.fuse_probabilities(prob_aml=aml_probs, prob_dl=dl_probs, alpha=0.0)
        np.testing.assert_allclose(fused, aml_probs)

    def test_pure_dl_fusion_alpha_one(self):
        """Alpha = 1.0 must exactly equal DL probabilities."""
        aml_probs = np.array([0.1, 0.4, 0.7, 0.9])
        dl_probs = np.array([0.3, 0.5, 0.2, 0.8])
        fused = dl_utils.fuse_probabilities(prob_aml=aml_probs, prob_dl=dl_probs, alpha=1.0)
        np.testing.assert_allclose(fused, dl_probs)

    def test_balanced_hybrid_fusion_alpha_half(self):
        """Alpha = 0.5 must equal arithmetic mean of AML and DL probabilities."""
        aml_probs = np.array([0.2, 0.6])
        dl_probs = np.array([0.4, 0.8])
        expected = np.array([0.3, 0.7])
        fused = dl_utils.fuse_probabilities(prob_aml=aml_probs, prob_dl=dl_probs, alpha=0.5)
        np.testing.assert_allclose(fused, expected)

    def test_intermediate_alpha_interpolation(self):
        """Intermediate alpha values (e.g. 0.35, 0.70) must correctly compute weighted sum."""
        aml_probs = np.array([0.20, 0.80])
        dl_probs = np.array([0.60, 0.40])
        # alpha = 0.35: 0.35 * DL + 0.65 * AML
        # [0.35*0.60 + 0.65*0.20, 0.35*0.40 + 0.65*0.80] = [0.21 + 0.13, 0.14 + 0.52] = [0.34, 0.66]
        fused = dl_utils.fuse_probabilities(prob_aml=aml_probs, prob_dl=dl_probs, alpha=0.35)
        np.testing.assert_allclose(fused, [0.34, 0.66])

    def test_invalid_alpha_raises_value_error(self):
        """Alpha outside [0, 1] must raise ValueError."""
        probs = np.array([0.5])
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(prob_aml=probs, prob_dl=probs, alpha=-0.1)
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(prob_aml=probs, prob_dl=probs, alpha=1.1)


class TestTask4LinguisticSanityChecks(unittest.TestCase):
    """Test linguistic sanity filtering rules, overlapping diagnostics, and missing data policy."""

    def test_negation_pattern_detects_cues(self):
        """Negation pattern must match explicit negation cues in intervening text."""
        neg_samples = [
            "does not activate",
            "failed to promote transcription",
            "neither inhibits nor stimulates",
            "unable to induce expression",
            "without affecting growth",
        ]
        for s in neg_samples:
            self.assertTrue(bool(NEGATION_PATTERN.search(s)), f"Failed to detect negation in: '{s}'")

    def test_negation_pattern_ignores_affirmative_text(self):
        """Negation pattern must NOT trigger on affirmative causal phrases."""
        pos_samples = [
            "strongly promotes transcription of",
            "directly activates the sensor kinase",
            "leads to increased resistance",
            "results in rapid upregulation",
        ]
        for s in pos_samples:
            self.assertFalse(bool(NEGATION_PATTERN.search(s)), f"False positive negation in: '{s}'")

    def test_linguistic_filtering_non_overlapping_diagnostics(self):
        """Verify auditable non-overlapping rejection categories: negation only, distance only, both."""
        df_test = pd.DataFrame({
            "candidate_id": ["c1", "c2", "c3", "c4", "c5"],
            "raw_pred": [1, 1, 1, 1, 0],
            "between_text": [
                "directly activates",                                   # Survives: neither
                "does not activate",                                    # Negation only
                "promotes " + "word " * 100,                           # Distance only (>400 chars)
                "does not promote " + "word " * 100,                   # Both negation and distance
                "neither inhibits nor activates",                      # Negative raw pred (ignored)
            ]
        })

        df_filtered = apply_linguistic_filters(df_test, max_between_chars=400, pred_col="raw_pred")

        pos_mask = df_test["raw_pred"] == 1
        pos_df = df_filtered[pos_mask]

        n_neg_only = int(((pos_df["flag_negation"]) & (~pos_df["flag_distance"])).sum())
        n_dist_only = int(((~pos_df["flag_negation"]) & (pos_df["flag_distance"])).sum())
        n_both = int(((pos_df["flag_negation"]) & (pos_df["flag_distance"])).sum())
        n_unique_rejected = int((pos_df["flag_negation"] | pos_df["flag_distance"]).sum())
        n_surviving = int((pos_df["sanitized_pred"] == 1).sum())

        self.assertEqual(n_neg_only, 1)
        self.assertEqual(n_dist_only, 1)
        self.assertEqual(n_both, 1)
        self.assertEqual(n_unique_rejected, 3)
        self.assertEqual(n_surviving, 1)

        # Invariant arithmetic: neg_only + dist_only + both == unique_rejected
        self.assertEqual(n_neg_only + n_dist_only + n_both, n_unique_rejected)
        # Invariant arithmetic: surviving == initial - unique_rejected
        self.assertEqual(n_surviving, int(pos_mask.sum()) - n_unique_rejected)

    def test_missing_between_text_raises_value_error(self):
        """Missing or NaN between_text must raise ValueError and never pass as string 'nan'."""
        df_bad = pd.DataFrame({
            "candidate_id": ["c1"],
            "raw_pred": [1],
            "between_text": [np.nan],
        })
        with self.assertRaises(ValueError) as ctx:
            apply_linguistic_filters(df_bad)
        self.assertIn("Missing between_text encountered", str(ctx.exception))


class TestTask4UnifiedRelationRecordSchema(unittest.TestCase):
    """Test 17-column UnifiedRelationRecord schema conformance."""

    def test_valid_record_matches_exact_schema(self):
        """Record formatted via dl_utils must have exactly the 17 specified keys in exact order."""
        rec = dl_utils.build_unified_relation_record(
            candidate_id="PMC1804205-00-TIAB_T29_T27",
            article_id="PMC1804205",
            section_id="PMC1804205-00-TIAB",
            cause_id="T29",
            cause_text="PmrB sensor kinase",
            cause_start=100,
            cause_end=118,
            effect_id="T27",
            effect_text="PmrA regulator",
            effect_start=200,
            effect_end=214,
            confidence=0.88,
            extraction_method="hybrid_probability_fusion",
            evidence_text="requires both PmrA's cognate sensor kinase PmrB",
            split="test",
            direction="cause->effect",
            provenance="PubMedBERT_alpha0.60+LogisticRegression_1to2",
        )
        self.assertEqual(list(rec.keys()), dl_utils.UNIFIED_RELATION_SCHEMA)
        self.assertEqual(len(rec), 17)
        self.assertEqual(rec["direction"], "cause->effect")
        self.assertTrue(rec["relation_id"].startswith("dl_"))
        self.assertGreaterEqual(rec["confidence"], 0.0)
        self.assertLessEqual(rec["confidence"], 1.0)


class TestTask4DAGAndBayesianNetwork(unittest.TestCase):
    """Test DAG construction, self-loop, bidirectional conflict, and cycle handling."""

    def test_enforce_dag_acyclicity_removes_self_loops(self):
        """Self-loops (u -> u) must be pruned by enforce_dag_acyclicity."""
        G = nx.DiGraph()
        G.add_edge("A", "A", confidence=0.9)
        G.add_edge("A", "B", confidence=0.8)

        G_dag, audit_log = enforce_dag_acyclicity(G, weight_attr="confidence")
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertFalse(G_dag.has_edge("A", "A"))
        self.assertTrue(G_dag.has_edge("A", "B"))
        self.assertTrue(any(e.get("reason") == "self_loop" for e in audit_log))

    def test_enforce_dag_acyclicity_resolves_bidirectional_conflicts(self):
        """Bidirectional conflicts (A -> B and B -> A) must prune the lower-confidence edge."""
        G = nx.DiGraph()
        G.add_edge("A", "B", confidence=0.85)
        G.add_edge("B", "A", confidence=0.45)  # Lower weight

        G_dag, audit_log = enforce_dag_acyclicity(G, weight_attr="confidence")
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertTrue(G_dag.has_edge("A", "B"))
        self.assertFalse(G_dag.has_edge("B", "A"))
        self.assertTrue(any("bidirectional" in e.get("reason", "") for e in audit_log))

    def test_enforce_dag_acyclicity_removes_feedback_cycles(self):
        """Cyclic graph (A -> B -> C -> A) must prune the lowest weight feedback edge."""
        G = nx.DiGraph()
        G.add_edge("A", "B", confidence=0.90)
        G.add_edge("B", "C", confidence=0.80)
        G.add_edge("C", "A", confidence=0.60)  # Lowest weight feedback edge

        self.assertFalse(nx.is_directed_acyclic_graph(G))
        G_dag, audit_log = enforce_dag_acyclicity(G, weight_attr="confidence")
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertTrue(G_dag.has_edge("A", "B"))
        self.assertTrue(G_dag.has_edge("B", "C"))
        self.assertFalse(G_dag.has_edge("C", "A"))
        self.assertTrue(any("cycle" in e.get("reason", "") for e in audit_log))

    def test_pgmpy_bayesian_network_structural_validity_and_inference(self):
        """Compiled acyclic graph must validate in pgmpy and evaluate observational vs interventional queries."""
        # Simple regulatory cascade: Sensor (S) -> Regulator (R) -> Target (T)
        bn = DiscreteBayesianNetwork([("S", "R"), ("R", "T")])
        self.assertTrue(nx.is_directed_acyclic_graph(bn))
        self.assertEqual(len(bn.nodes()), 3)
        self.assertEqual(len(bn.edges()), 2)

        cpd_s = TabularCPD(variable="S", variable_card=2, values=[[0.7], [0.3]], state_names={"S": ["0", "1"]})
        cpd_r = TabularCPD(
            variable="R", variable_card=2,
            values=[[0.85, 0.20], [0.15, 0.80]],
            evidence=["S"], evidence_card=[2],
            state_names={"R": ["0", "1"], "S": ["0", "1"]}
        )
        cpd_t = TabularCPD(
            variable="T", variable_card=2,
            values=[[0.90, 0.10], [0.10, 0.90]],
            evidence=["R"], evidence_card=[2],
            state_names={"T": ["0", "1"], "R": ["0", "1"]}
        )
        bn.add_cpds(cpd_s, cpd_r, cpd_t)
        self.assertTrue(bn.check_model())

        infer = VariableElimination(bn)
        # Observational query: P(T=1 | S=1)
        q_obs = infer.query(["T"], evidence={"S": "1"})
        p_obs = float(q_obs.values[1])

        # Interventional query via graph mutilation: P(T=1 | do(R=1))
        interv_bn = bn.do(["R"])
        infer_interv = VariableElimination(interv_bn)
        q_interv = infer_interv.query(["T"], evidence={"R": "1"})
        p_interv = float(q_interv.values[1])

        self.assertGreater(p_obs, 0.0)
        self.assertGreater(p_interv, 0.0)
        self.assertNotEqual(p_obs, p_interv)


if __name__ == "__main__":
    unittest.main()
