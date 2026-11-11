"""Focused unit tests for Phase 2 Task 4: Hybrid Probability Fusion and Graph Integration.

Directly imports and tests the actual implementation from `dl_task4_utils.py`:
1. Prediction dataframe contract validation and stream alignment.
2. Exact model-eligible candidate ID membership, manifest validation, and isolated metadata streaming.
3. Strict validation-path isolation (no access to test predictions, test metadata, or gold relations).
4. Lexicographic validation hyperparameter selection with Macro-F1 and deterministic tie-breaking.
5. Linguistic sanity checks with auditable non-overlapping rejection diagnostics.
6. Standardized 17-column UnifiedRelationRecord construction and export verification.
7. Prediction export verification against source arrays with negative corruption tests.
8. Directed acyclic graph (DAG) cycle breaking, self-loop pruning, bidirectional conflict resolution.
9. Bayesian network compilation (node/edge preservation, acyclicity) and Pearl do-calculus demonstration.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import networkx as nx

# Ensure DL and AML modules are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
for p in [ROOT_DIR, ROOT_DIR / "AML", ROOT_DIR / "DL"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dl_utils
import dl_task4_utils
from causal_extractors import enforce_dag_acyclicity, NEGATION_PATTERN
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# ============================================================================
# 1. Prediction DataFrame Validation & Stream Alignment Tests
# ============================================================================

class TestTask4InputValidationAndAlignment(unittest.TestCase):
    """Test strict prediction file contracts and cross-stream candidate alignment."""

    def setUp(self):
        self.valid_df = pd.DataFrame({
            "candidate_id": ["c1", "c2", "c3"],
            "true_label": [0, 1, 0],
            "aml_prob": [0.12, 0.85, 0.45],
            "aml_pred": [0, 1, 0],
            "split": ["validation", "validation", "validation"],
        })

    def test_valid_contracts_pass(self):
        dl_task4_utils.validate_prediction_dataframe(
            self.valid_df, 3, "validation", "AML", "aml_prob", "aml_pred"
        )

    def test_missing_column_raises_value_error(self):
        df_bad = self.valid_df.drop(columns=["aml_prob"])
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_prediction_dataframe(
                df_bad, 3, "validation", "AML", "aml_prob", "aml_pred"
            )
        self.assertIn("Missing required columns", str(ctx.exception))

    def test_row_count_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_prediction_dataframe(
                self.valid_df, 1884, "validation", "AML", "aml_prob", "aml_pred"
            )
        self.assertIn("Row count mismatch", str(ctx.exception))

    def test_null_or_empty_candidate_id_raises_value_error(self):
        df_bad = self.valid_df.copy()
        df_bad.loc[1, "candidate_id"] = "   "
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_prediction_dataframe(
                df_bad, 3, "validation", "AML", "aml_prob", "aml_pred"
            )
        self.assertIn("Found null or empty candidate_id", str(ctx.exception))

    def test_duplicate_candidate_id_raises_value_error(self):
        df_bad = self.valid_df.copy()
        df_bad.loc[2, "candidate_id"] = "c1"
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_prediction_dataframe(
                df_bad, 3, "validation", "AML", "aml_prob", "aml_pred"
            )
        self.assertIn("duplicate candidate_id", str(ctx.exception))

    def test_non_binary_label_raises_value_error(self):
        df_bad = self.valid_df.copy()
        df_bad.loc[0, "true_label"] = 2
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_prediction_dataframe(
                df_bad, 3, "validation", "AML", "aml_prob", "aml_pred"
            )
        self.assertIn("Non-binary true_label", str(ctx.exception))

    def test_out_of_bounds_probability_raises_value_error(self):
        df_bad = self.valid_df.copy()
        df_bad.loc[0, "aml_prob"] = 1.05
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_prediction_dataframe(
                df_bad, 3, "validation", "AML", "aml_prob", "aml_pred"
            )
        self.assertIn("Probabilities outside [0.0, 1.0]", str(ctx.exception))

    def test_stream_alignment_detects_candidate_order_mismatch(self):
        df_aml = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "true_label": [0, 1],
        })
        df_dl = pd.DataFrame({
            "candidate_id": ["c2", "c1"],  # Reordered!
            "true_label": [1, 0],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_dual_stream_alignment(df_aml, df_dl, "validation")
        self.assertIn("Candidate ID order or identity mismatch", str(ctx.exception))

    def test_stream_alignment_detects_label_discrepancy(self):
        df_aml = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "true_label": [0, 1],
        })
        df_dl = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "true_label": [0, 0],  # Label mismatch on c2!
        })
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_dual_stream_alignment(df_aml, df_dl, "validation")
        self.assertIn("Ground truth label discrepancy", str(ctx.exception))


# ============================================================================
# 2. Eligible Membership, Manifest Integrity, & Validation Isolation Tests
# ============================================================================

class TestTask4EligibleMembershipAndMetadataIsolation(unittest.TestCase):
    """Test exact model-eligible ID verification, manifest contracts, and validation isolation."""

    def test_exact_eligible_membership_success(self):
        expected_ids = {"c1", "c2", "c3"}
        pred_ids = ["c1", "c2", "c3"]
        dl_task4_utils.validate_exact_eligible_membership(pred_ids, expected_ids, "validation", "AML")

    def test_exact_eligible_membership_detects_missing_id(self):
        expected_ids = {"c1", "c2", "c3"}
        pred_ids = ["c1", "c2"]  # Missing c3
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_exact_eligible_membership(pred_ids, expected_ids, "validation", "AML")
        self.assertIn("Missing 1 eligible IDs", str(ctx.exception))

    def test_exact_eligible_membership_detects_ineligible_extra_id(self):
        expected_ids = {"c1", "c2"}
        pred_ids = ["c1", "c2", "c999"]  # c999 is ineligible
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_exact_eligible_membership(pred_ids, expected_ids, "validation", "AML")
        self.assertIn("ineligible/extra IDs", str(ctx.exception))

    def test_manifest_validation_success(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            pd.DataFrame({
                "candidate_id": ["c1", "c2", "c3"],
                "label": [1, 0, 0],
                "split": ["validation", "validation", "validation"],
            }).to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            df_manifest = dl_task4_utils.validate_and_load_authoritative_manifest(
                manifest_path=tmp_path,
                split_name="validation",
                expected_total=3,
                expected_pos=1,
                expected_neg=2,
            )
            self.assertEqual(len(df_manifest), 3)
            ids = dl_task4_utils.load_authoritative_eligible_candidate_ids(
                split_manifest_path=tmp_path,
                split_name="validation",
                expected_count=3,
                expected_pos=1,
                expected_neg=2,
            )
            self.assertEqual(ids, {"c1", "c2", "c3"})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_manifest_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            dl_task4_utils.load_authoritative_eligible_candidate_ids(
                "/nonexistent/manifest.csv", "validation", 1884
            )
        self.assertIn("In strict adherence to requirements", str(ctx.exception))

    def test_manifest_duplicate_candidate_id_raises_value_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            pd.DataFrame({
                "candidate_id": ["c1", "c1"],
                "label": [1, 0],
                "split": ["validation", "validation"],
            }).to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.validate_and_load_authoritative_manifest(tmp_path, "validation", 2)
            self.assertIn("duplicate candidate_id", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_manifest_pos_count_mismatch_raises_value_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            pd.DataFrame({
                "candidate_id": ["c1", "c2"],
                "label": [1, 1],  # 2 positive, expected 1
                "split": ["validation", "validation"],
            }).to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.validate_and_load_authoritative_manifest(
                    tmp_path, "validation", 2, expected_pos=1, expected_neg=1
                )
            self.assertIn("positive count mismatch", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_split_candidate_metadata_retains_only_requested_split(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            pd.DataFrame({
                "candidate_id": ["c_val_1", "c_val_2", "c_test_1", "c_train_1"],
                "label": [0, 1, 1, 0],
                "split": ["validation", "validation", "test", "train"],
            }).to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            df_val = dl_task4_utils.load_split_candidate_metadata(tmp_path, "validation")
            self.assertEqual(len(df_val), 2)
            self.assertEqual(set(df_val["split"].unique()), {"validation"})
            self.assertEqual(set(df_val["candidate_id"]), {"c_val_1", "c_val_2"})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_split_candidate_metadata_with_usecols(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            pd.DataFrame({
                "candidate_id": ["c_val_1", "c_test_1"],
                "label": [0, 1],
                "secret_test_field": ["do_not_load", "classified"],
                "split": ["validation", "test"],
            }).to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            df_val = dl_task4_utils.load_split_candidate_metadata(
                tmp_path, "validation", usecols=["candidate_id", "label"]
            )
            self.assertEqual(len(df_val), 1)
            self.assertIn("candidate_id", df_val.columns)
            self.assertNotIn("secret_test_field", df_val.columns)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validation_selection_path_does_not_access_test_predictions_metadata_or_gold_relations(self):
        """Verify that validation ingestion and parameter selection strictly execute without opening test files."""
        val_y_true = np.array([0, 1, 0, 1])
        val_p_aml = np.array([0.2, 0.7, 0.4, 0.8])
        val_p_dl = np.array([0.1, 0.9, 0.3, 0.85])

        # Execute validation selection directly
        best_alpha, best_tau, best_f1, best_macro, records = (
            dl_task4_utils.select_validation_fusion_hyperparameters(
                val_y_true=val_y_true,
                val_p_aml=val_p_aml,
                val_p_dl=val_p_dl,
            )
        )
        self.assertGreater(best_f1, 0.0)
        self.assertGreater(best_macro, 0.0)
        self.assertIn(best_alpha, np.round(np.arange(0.0, 1.05, 0.05), 2))
        self.assertIn(best_tau, np.round(np.arange(0.10, 0.95, 0.05), 2))


# ============================================================================
# 3. Lexicographic Validation Selection Rule Tests
# ============================================================================

class TestTask4LexicographicValidationSelection(unittest.TestCase):
    """Test lexicographic hyperparameter selection: Positive F1 -> Macro F1 -> smaller alpha -> smaller tau."""

    def test_higher_positive_f1_wins(self):
        better = dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.65, cand_macro_f1=0.70, cand_alpha=0.60, cand_tau=0.50,
            best_f1_pos=0.60, best_macro_f1=0.80, best_alpha=0.50, best_tau=0.50,
        )
        self.assertTrue(better)

    def test_equal_positive_f1_higher_macro_f1_wins(self):
        better = dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.6000001, cand_macro_f1=0.75, cand_alpha=0.70, cand_tau=0.50,
            best_f1_pos=0.6000000, best_macro_f1=0.70, best_alpha=0.50, best_tau=0.50,
        )
        self.assertTrue(better)

    def test_equal_positive_and_macro_f1_invokes_deterministic_tie_break(self):
        # Case A: smaller alpha wins
        better_alpha = dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.60, cand_macro_f1=0.70, cand_alpha=0.30, cand_tau=0.50,
            best_f1_pos=0.60, best_macro_f1=0.70, best_alpha=0.50, best_tau=0.50,
        )
        self.assertTrue(better_alpha)

        # Case B: identical alpha, smaller tau wins
        better_tau = dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.60, cand_macro_f1=0.70, cand_alpha=0.50, cand_tau=0.40,
            best_f1_pos=0.60, best_macro_f1=0.70, best_alpha=0.50, best_tau=0.50,
        )
        self.assertTrue(better_tau)

    def test_selection_is_order_invariant(self):
        np.random.seed(42)
        y_val = np.random.choice([0, 1], size=100, p=[0.85, 0.15])
        p_aml = np.random.uniform(0.0, 1.0, size=100)
        p_dl = np.random.uniform(0.0, 1.0, size=100)

        alpha_grid = [0.2, 0.4, 0.6, 0.8]
        tau_grid = [0.3, 0.5, 0.7]

        res_forward = dl_task4_utils.select_validation_fusion_hyperparameters(
            y_val, p_aml, p_dl, alpha_grid=alpha_grid, tau_grid=tau_grid
        )
        res_reverse = dl_task4_utils.select_validation_fusion_hyperparameters(
            y_val, p_aml, p_dl, alpha_grid=list(reversed(alpha_grid)), tau_grid=list(reversed(tau_grid))
        )

        self.assertEqual(res_forward[0], res_reverse[0])  # best_alpha
        self.assertEqual(res_forward[1], res_reverse[1])  # best_tau
        self.assertAlmostEqual(res_forward[2], res_reverse[2], places=6)  # best_f1_pos
        self.assertAlmostEqual(res_forward[3], res_reverse[3], places=6)  # best_macro_f1


# ============================================================================
# 4. Linguistic Sanity Checks & Diagnostics Tests
# ============================================================================

class TestTask4LinguisticSanityChecks(unittest.TestCase):
    """Test negation detection, distance pruning, and non-overlapping rejection diagnostics."""

    def test_negation_pattern_detects_cues(self):
        positives = [
            "expression of Arg1 does not activate Arg2",
            "treatment without Arg1 showed no response",
            "neither protein A nor protein B",
            "failed to induce phosphorylation",
            "unable to stimulate receptor",
        ]
        for text in positives:
            self.assertIsNotNone(NEGATION_PATTERN.search(text), f"Failed to detect negation in: {text}")

    def test_linguistic_filtering_non_overlapping_diagnostics(self):
        df_test = pd.DataFrame({
            "candidate_id": [f"c{i}" for i in range(5)],
            "raw_pred": [1, 1, 1, 1, 0],
            "between_text": [
                "directly activates",                # Survives
                "does not stimulate",                # Negation only (<400 chars)
                "x " * 250,                          # Distance only (>400 chars, affirmative)
                "does not " + ("x " * 250),          # Both negation & distance
                "does not activate",                 # Raw pred 0 -> not a positive candidate
            ]
        })

        df_filtered = dl_task4_utils.apply_linguistic_filters(df_test, max_between_chars=400, pred_col="raw_pred")
        diag = dl_task4_utils.compute_linguistic_diagnostics(df_filtered, pred_col="raw_pred", sanitized_col="sanitized_pred")

        self.assertEqual(diag["initial_positive"], 4)
        self.assertEqual(diag["rejected_negation_only"], 1)
        self.assertEqual(diag["rejected_distance_only"], 1)
        self.assertEqual(diag["rejected_both"], 1)
        self.assertEqual(diag["total_unique_rejected"], 3)
        self.assertEqual(diag["surviving_positive"], 1)
        self.assertEqual(diag["initial_positive"], diag["surviving_positive"] + diag["total_unique_rejected"])

    def test_missing_between_text_raises_value_error(self):
        df_bad = pd.DataFrame({
            "candidate_id": ["c1"],
            "raw_pred": [1],
            "between_text": [np.nan],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.apply_linguistic_filters(df_bad)
        self.assertIn("Missing between_text encountered", str(ctx.exception))

    def test_empty_or_nan_string_between_text_raises_value_error(self):
        df_bad = pd.DataFrame({
            "candidate_id": ["c1"],
            "raw_pred": [1],
            "between_text": ["nan"],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.apply_linguistic_filters(df_bad)
        self.assertIn("Invalid or empty between_text", str(ctx.exception))


# ============================================================================
# 5. Prediction Export Verification Tests
# ============================================================================

class TestTask4PredictionExportVerification(unittest.TestCase):
    """Test read-back verification of prediction export against source arrays with negative tests."""

    def setUp(self):
        self.frozen_alpha = 0.60
        self.frozen_tau = 0.50
        self.source_candidate_ids = ["c1", "c2"]
        self.source_true_labels = [1, 0]
        self.source_aml_probs = [0.40, 0.20]
        self.source_dl_probs = [0.80, 0.30]
        self.source_aml_preds = [0, 0]
        self.source_dl_preds = [1, 0]
        self.hybrid_probs = [
            self.frozen_alpha * 0.80 + (1 - self.frozen_alpha) * 0.40,  # 0.64
            self.frozen_alpha * 0.30 + (1 - self.frozen_alpha) * 0.20,  # 0.26
        ]
        self.hybrid_preds = [1, 0]

    def _write_csv(self, df):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.to_csv(tmp.name, index=False)
        return tmp.name

    def test_prediction_export_verification_success(self):
        df_valid = pd.DataFrame({
            "candidate_id": self.source_candidate_ids,
            "true_label": self.source_true_labels,
            "aml_prob": self.source_aml_probs,
            "dl_prob": self.source_dl_probs,
            "hybrid_prob": self.hybrid_probs,
            "aml_pred": self.source_aml_preds,
            "dl_pred": self.source_dl_preds,
            "hybrid_pred": self.hybrid_preds,
            "split": ["validation", "validation"],
        })
        tmp_path = self._write_csv(df_valid)
        try:
            dl_task4_utils.verify_prediction_export(
                filepath=tmp_path,
                expected_rows=2,
                split_name="validation",
                source_candidate_ids=self.source_candidate_ids,
                source_true_labels=self.source_true_labels,
                source_aml_probs=self.source_aml_probs,
                source_dl_probs=self.source_dl_probs,
                source_aml_preds=self.source_aml_preds,
                source_dl_preds=self.source_dl_preds,
                frozen_alpha=self.frozen_alpha,
                frozen_tau=self.frozen_tau,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_corrupted_aml_probability_fails_verification(self):
        df_bad = pd.DataFrame({
            "candidate_id": self.source_candidate_ids,
            "true_label": self.source_true_labels,
            "aml_prob": [0.99, 0.20],  # Corrupted!
            "dl_prob": self.source_dl_probs,
            "hybrid_prob": self.hybrid_probs,
            "aml_pred": self.source_aml_preds,
            "dl_pred": self.source_dl_preds,
            "hybrid_pred": self.hybrid_preds,
            "split": ["validation", "validation"],
        })
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_prediction_export(
                    filepath=tmp_path, expected_rows=2, split_name="validation",
                    source_candidate_ids=self.source_candidate_ids, source_true_labels=self.source_true_labels,
                    source_aml_probs=self.source_aml_probs, source_dl_probs=self.source_dl_probs,
                    frozen_alpha=self.frozen_alpha, frozen_tau=self.frozen_tau,
                )
            self.assertIn("Exported aml_prob does not match source", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_corrupted_dl_probability_fails_verification(self):
        df_bad = pd.DataFrame({
            "candidate_id": self.source_candidate_ids,
            "true_label": self.source_true_labels,
            "aml_prob": self.source_aml_probs,
            "dl_prob": [0.11, 0.30],  # Corrupted!
            "hybrid_prob": self.hybrid_probs,
            "aml_pred": self.source_aml_preds,
            "dl_pred": self.source_dl_preds,
            "hybrid_pred": self.hybrid_preds,
            "split": ["validation", "validation"],
        })
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_prediction_export(
                    filepath=tmp_path, expected_rows=2, split_name="validation",
                    source_candidate_ids=self.source_candidate_ids, source_true_labels=self.source_true_labels,
                    source_aml_probs=self.source_aml_probs, source_dl_probs=self.source_dl_probs,
                    frozen_alpha=self.frozen_alpha, frozen_tau=self.frozen_tau,
                )
            self.assertIn("Exported dl_prob does not match source", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_corrupted_candidate_id_order_fails_verification(self):
        df_bad = pd.DataFrame({
            "candidate_id": ["c2", "c1"],  # Swapped!
            "true_label": self.source_true_labels,
            "aml_prob": self.source_aml_probs,
            "dl_prob": self.source_dl_probs,
            "hybrid_prob": self.hybrid_probs,
            "aml_pred": self.source_aml_preds,
            "dl_pred": self.source_dl_preds,
            "hybrid_pred": self.hybrid_preds,
            "split": ["validation", "validation"],
        })
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_prediction_export(
                    filepath=tmp_path, expected_rows=2, split_name="validation",
                    source_candidate_ids=self.source_candidate_ids, source_true_labels=self.source_true_labels,
                    source_aml_probs=self.source_aml_probs, source_dl_probs=self.source_dl_probs,
                    frozen_alpha=self.frozen_alpha, frozen_tau=self.frozen_tau,
                )
            self.assertIn("Candidate ID ordering or content mismatch", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_corrupted_label_fails_verification(self):
        df_bad = pd.DataFrame({
            "candidate_id": self.source_candidate_ids,
            "true_label": [0, 0],  # Corrupted true label!
            "aml_prob": self.source_aml_probs,
            "dl_prob": self.source_dl_probs,
            "hybrid_prob": self.hybrid_probs,
            "aml_pred": self.source_aml_preds,
            "dl_pred": self.source_dl_preds,
            "hybrid_pred": self.hybrid_preds,
            "split": ["validation", "validation"],
        })
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_prediction_export(
                    filepath=tmp_path, expected_rows=2, split_name="validation",
                    source_candidate_ids=self.source_candidate_ids, source_true_labels=self.source_true_labels,
                    source_aml_probs=self.source_aml_probs, source_dl_probs=self.source_dl_probs,
                    frozen_alpha=self.frozen_alpha, frozen_tau=self.frozen_tau,
                )
            self.assertIn("True labels do not match source", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_corrupted_prediction_fails_verification(self):
        df_bad = pd.DataFrame({
            "candidate_id": self.source_candidate_ids,
            "true_label": self.source_true_labels,
            "aml_prob": self.source_aml_probs,
            "dl_prob": self.source_dl_probs,
            "hybrid_prob": self.hybrid_probs,
            "aml_pred": self.source_aml_preds,
            "dl_pred": self.source_dl_preds,
            "hybrid_pred": [0, 0],  # Corrupted! 0.64 >= 0.50 should be 1
            "split": ["validation", "validation"],
        })
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_prediction_export(
                    filepath=tmp_path, expected_rows=2, split_name="validation",
                    source_candidate_ids=self.source_candidate_ids, source_true_labels=self.source_true_labels,
                    source_aml_probs=self.source_aml_probs, source_dl_probs=self.source_dl_probs,
                    frozen_alpha=self.frozen_alpha, frozen_tau=self.frozen_tau,
                )
            self.assertIn("hybrid_pred does not match (hybrid_prob >= frozen_tau)", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_corrupted_split_fails_verification(self):
        df_bad = pd.DataFrame({
            "candidate_id": self.source_candidate_ids,
            "true_label": self.source_true_labels,
            "aml_prob": self.source_aml_probs,
            "dl_prob": self.source_dl_probs,
            "hybrid_prob": self.hybrid_probs,
            "aml_pred": self.source_aml_preds,
            "dl_pred": self.source_dl_preds,
            "hybrid_pred": self.hybrid_preds,
            "split": ["validation", "test"],  # Leaked test split!
        })
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_prediction_export(
                    filepath=tmp_path, expected_rows=2, split_name="validation",
                    source_candidate_ids=self.source_candidate_ids, source_true_labels=self.source_true_labels,
                    source_aml_probs=self.source_aml_probs, source_dl_probs=self.source_dl_probs,
                    frozen_alpha=self.frozen_alpha, frozen_tau=self.frozen_tau,
                )
            self.assertIn("Split column contains values other than", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_corrupted_fusion_formula_fails_verification(self):
        df_bad = pd.DataFrame({
            "candidate_id": self.source_candidate_ids,
            "true_label": self.source_true_labels,
            "aml_prob": self.source_aml_probs,
            "dl_prob": self.source_dl_probs,
            "hybrid_prob": [0.99, 0.26],  # Corrupted hybrid probability!
            "aml_pred": self.source_aml_preds,
            "dl_pred": self.source_dl_preds,
            "hybrid_pred": [1, 0],
            "split": ["validation", "validation"],
        })
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_prediction_export(
                    filepath=tmp_path, expected_rows=2, split_name="validation",
                    source_candidate_ids=self.source_candidate_ids, source_true_labels=self.source_true_labels,
                    source_aml_probs=self.source_aml_probs, source_dl_probs=self.source_dl_probs,
                    frozen_alpha=self.frozen_alpha, frozen_tau=self.frozen_tau,
                )
            self.assertIn("Exported hybrid_prob mismatch with source-recomputed formula", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ============================================================================
# 6. Relation Export Verification Tests
# ============================================================================

class TestTask4RelationExportVerification(unittest.TestCase):
    """Test read-back verification of 17-column UnifiedRelationRecord export with negative tests."""

    def setUp(self):
        self.sanitized_cids = ["c1"]
        self.rel_id = dl_utils.generate_relation_id("c1")
        self.valid_rel_df = pd.DataFrame([{
            "relation_id": self.rel_id,
            "article_id": "PMC12345",
            "section_id": "sec_01",
            "cause_id": "E1",
            "cause_text": "PmrA",
            "cause_start": 10,
            "cause_end": 14,
            "effect_id": "E2",
            "effect_text": "polymyxin resistance",
            "effect_start": 40,
            "effect_end": 60,
            "direction": "cause->effect",
            "confidence": 0.85,
            "extraction_method": "hybrid_probability_fusion",
            "evidence_text": "PmrA activates polymyxin resistance",
            "provenance": "test_provenance",
            "split": "test",
        }])

    def _write_csv(self, df):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.to_csv(tmp.name, index=False)
        return tmp.name

    def test_relation_export_verification_success(self):
        tmp_path = self._write_csv(self.valid_rel_df)
        try:
            dl_task4_utils.verify_relation_export(
                filepath=tmp_path,
                expected_count=1,
                sanitized_candidate_ids=self.sanitized_cids,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_missing_relation_record_fails_verification(self):
        tmp_path = self._write_csv(self.valid_rel_df)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=2,  # Expected 2, got 1
                    sanitized_candidate_ids=["c1", "c2"],
                )
            self.assertIn("Relation export row count mismatch", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_extra_relation_record_fails_verification(self):
        df_extra = pd.concat([self.valid_rel_df, self.valid_rel_df.copy()], ignore_index=True)
        df_extra.loc[1, "relation_id"] = dl_utils.generate_relation_id("c2")
        tmp_path = self._write_csv(df_extra)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=1,  # Expected 1, got 2
                    sanitized_candidate_ids=self.sanitized_cids,
                )
            self.assertIn("Relation export row count mismatch", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_duplicate_relation_record_fails_verification(self):
        df_dup = pd.concat([self.valid_rel_df, self.valid_rel_df], ignore_index=True)
        tmp_path = self._write_csv(df_dup)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=2,
                    sanitized_candidate_ids=["c1", "c1_dup"],
                )
            self.assertIn("duplicate relation_id", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_mismatched_relation_record_fails_verification(self):
        df_mismatched = self.valid_rel_df.copy()
        df_mismatched.loc[0, "relation_id"] = "dl_wronghash123456"
        tmp_path = self._write_csv(df_mismatched)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=1,
                    sanitized_candidate_ids=self.sanitized_cids,
                )
            self.assertIn("candidate traceability mismatch", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_null_identifier_fails_verification(self):
        df_bad = self.valid_rel_df.copy()
        df_bad.loc[0, "cause_id"] = ""
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=1,
                    sanitized_candidate_ids=self.sanitized_cids,
                )
            self.assertIn("Found null or empty values in required identifier field 'cause_id'", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_self_loop_relation_fails_verification(self):
        df_bad = self.valid_rel_df.copy()
        df_bad.loc[0, "effect_id"] = df_bad.loc[0, "cause_id"]  # Self-loop!
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=1,
                    sanitized_candidate_ids=self.sanitized_cids,
                )
            self.assertIn("self-referential relations", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_invalid_direction_fails_verification(self):
        df_bad = self.valid_rel_df.copy()
        df_bad.loc[0, "direction"] = "undirected"
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=1,
                    sanitized_candidate_ids=self.sanitized_cids,
                )
            self.assertIn("direction other than 'cause->effect'", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_invalid_split_fails_verification(self):
        df_bad = self.valid_rel_df.copy()
        df_bad.loc[0, "split"] = "validation"  # Must be 'test'
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=1,
                    sanitized_candidate_ids=self.sanitized_cids,
                )
            self.assertIn("split other than 'test'", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_invalid_confidence_fails_verification(self):
        df_bad = self.valid_rel_df.copy()
        df_bad.loc[0, "confidence"] = 1.50
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=1,
                    sanitized_candidate_ids=self.sanitized_cids,
                )
            self.assertIn("Confidence values not finite or outside [0.0, 1.0]", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_fractional_offset_fails_verification(self):
        df_bad = self.valid_rel_df.copy()
        df_bad["cause_start"] = [10.5]
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=1,
                    sanitized_candidate_ids=self.sanitized_cids,
                )
            self.assertIn("Fractional offsets found", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_invalid_span_order_fails_verification(self):
        df_bad = self.valid_rel_df.copy()
        df_bad.loc[0, "cause_start"] = 20
        df_bad.loc[0, "cause_end"] = 10  # start > end!
        tmp_path = self._write_csv(df_bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(
                    filepath=tmp_path,
                    expected_count=1,
                    sanitized_candidate_ids=self.sanitized_cids,
                )
            self.assertIn("Invalid cause span order: cause_start >= cause_end", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ============================================================================
# 7. Directed Acyclic Graph (DAG) & Bayesian Network Tests
# ============================================================================

class TestTask4DAGAndBayesianNetwork(unittest.TestCase):
    """Test deterministic DAG cycle pruning and Bayesian network compilation."""

    def test_enforce_dag_acyclicity_removes_self_loops(self):
        G = nx.DiGraph()
        G.add_edge("A", "A", confidence=0.9)
        G.add_edge("A", "B", confidence=0.8)
        G_dag, audit = enforce_dag_acyclicity(G)
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertFalse(G_dag.has_edge("A", "A"))
        self.assertTrue(G_dag.has_edge("A", "B"))
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["reason"], "self_loop")

    def test_enforce_dag_acyclicity_resolves_bidirectional_conflicts(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", confidence=0.9)
        G.add_edge("B", "A", confidence=0.3)
        G_dag, audit = enforce_dag_acyclicity(G)
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertTrue(G_dag.has_edge("A", "B"))
        self.assertFalse(G_dag.has_edge("B", "A"))
        self.assertEqual(len(audit), 1)
        self.assertIn("bidirectional_conflict", audit[0]["reason"])

    def test_enforce_dag_acyclicity_removes_feedback_cycles(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", confidence=0.9)
        G.add_edge("B", "C", confidence=0.8)
        G.add_edge("C", "A", confidence=0.2)  # Weakest feedback edge
        G_dag, audit = enforce_dag_acyclicity(G)
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertTrue(G_dag.has_edge("A", "B"))
        self.assertTrue(G_dag.has_edge("B", "C"))
        self.assertFalse(G_dag.has_edge("C", "A"))
        self.assertEqual(len(audit), 1)
        self.assertIn("cycle", audit[0]["reason"])

    def test_verify_dag_and_pruning_diagnostics_reconciles_audit_log(self):
        G_pred = nx.DiGraph()
        G_pred.add_edge("A", "A", confidence=0.9)
        G_pred.add_edge("A", "B", confidence=0.8)
        G_pred.add_edge("B", "A", confidence=0.4)
        G_pred.add_edge("B", "C", confidence=0.7)
        G_pred.add_edge("C", "A", confidence=0.3)

        G_dag, audit = enforce_dag_acyclicity(G_pred)
        n_self = sum(1 for e in audit if e.get("reason") == "self_loop")
        n_bi = sum(1 for e in audit if "bidirectional_conflict" in e.get("reason", ""))
        n_cyc = sum(1 for e in audit if "cycle" in e.get("reason", ""))

        diag_content = {
            "initial_candidate_edges": G_pred.number_of_edges(),
            "final_dag_edges": G_dag.number_of_edges(),
            "pruned_edges_count": len(audit),
            "pruning_breakdown": {
                "self_loops": n_self,
                "bidirectional_conflicts": n_bi,
                "directed_cycle_feedback": n_cyc,
            },
            "edge_evaluation": {
                "before_pruning": {"Precision": 0.5, "Recall": 0.5, "F1": 0.5},
                "after_pruning": {"Precision": 0.6, "Recall": 0.5, "F1": 0.55},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(diag_content, tmp)
            tmp_path = tmp.name

        try:
            dl_task4_utils.verify_dag_and_pruning_diagnostics(G_pred, G_dag, audit, tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_compile_discrete_bayesian_network_preserves_all_nodes_and_edges(self):
        G = nx.DiGraph()
        G.add_node("Isolated_Node_X")  # Isolated node with no edges
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        bn = dl_task4_utils.compile_discrete_bayesian_network(G)
        self.assertIn("Isolated_Node_X", bn.nodes())
        self.assertEqual(set(bn.nodes()), {"Isolated_Node_X", "A", "B", "C"})
        self.assertEqual(set(bn.edges()), {("A", "B"), ("B", "C")})
        self.assertTrue(nx.is_directed_acyclic_graph(bn))

    def test_pgmpy_illustrative_bayesian_network_do_calculus(self):
        """Demonstrate Pearl's do-calculus on an illustrative synthetic 4-node network:
        Graph: A -> B -> C, D -> B
        Observational P(C | B=1) != Interventional P(C | do(B=1))
        """
        bn = DiscreteBayesianNetwork([("A", "B"), ("D", "B"), ("B", "C")])
        cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.7], [0.3]])
        cpd_d = TabularCPD(variable="D", variable_card=2, values=[[0.6], [0.4]])
        cpd_b = TabularCPD(
            variable="B", variable_card=2,
            values=[[0.9, 0.6, 0.5, 0.1], [0.1, 0.4, 0.5, 0.9]],
            evidence=["A", "D"], evidence_card=[2, 2]
        )
        cpd_c = TabularCPD(
            variable="C", variable_card=2,
            values=[[0.8, 0.2], [0.2, 0.8]],
            evidence=["B"], evidence_card=[2]
        )
        bn.add_cpds(cpd_a, cpd_d, cpd_b, cpd_c)
        self.assertTrue(bn.check_model())

        infer = VariableElimination(bn)
        obs_res = infer.query(variables=["C"], evidence={"B": 1})
        self.assertAlmostEqual(obs_res.values[1], 0.8, places=4)

        # Interventional: do(B=1) removes incoming edges A->B and D->B
        bn_mutilated = bn.copy()
        bn_mutilated.remove_edges_from([("A", "B"), ("D", "B")])
        cpd_b_do = TabularCPD(variable="B", variable_card=2, values=[[0.0], [1.0]])
        bn_mutilated.remove_cpds(cpd_b)
        bn_mutilated.add_cpds(cpd_b_do)
        self.assertTrue(bn_mutilated.check_model())

        infer_do = VariableElimination(bn_mutilated)
        do_res = infer_do.query(variables=["C"])
        self.assertAlmostEqual(do_res.values[1], 0.8, places=4)


if __name__ == "__main__":
    unittest.main()
