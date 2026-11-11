"""Focused unit tests for Phase 2 Task 4: Hybrid Probability Fusion and Graph Integration.

Directly imports and tests the actual implementation from `dl_task4_utils.py`:
1. Prediction dataframe contract validation and stream alignment.
2. Exact model-eligible candidate ID membership and isolated split metadata streaming.
3. Lexicographic validation hyperparameter selection with Macro-F1 and deterministic tie-breaking.
4. Linguistic sanity checks with auditable non-overlapping rejection diagnostics.
5. Standardized 17-column UnifiedRelationRecord construction and export verification.
6. Prediction export verification (schema, IDs, labels, recomputed fusion formula, binary predictions).
7. Directed acyclic graph (DAG) cycle breaking, self-loop pruning, bidirectional conflict resolution.
8. Bayesian network compilation (node/edge preservation, acyclicity) and Pearl do-calculus demonstration.
"""

import json
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
        self.assertIn("Probabilities outside", str(ctx.exception))

    def test_stream_alignment_detects_candidate_order_mismatch(self):
        df_aml = self.valid_df
        df_dl = self.valid_df.copy().iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_dual_stream_alignment(df_aml, df_dl, "Validation")
        self.assertIn("Candidate ID order or identity mismatch", str(ctx.exception))

    def test_stream_alignment_detects_label_discrepancy(self):
        df_aml = self.valid_df.copy()
        df_dl = self.valid_df.copy()
        df_dl.loc[0, "true_label"] = 1
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_dual_stream_alignment(df_aml, df_dl, "Validation")
        self.assertIn("Ground truth label discrepancy", str(ctx.exception))


# ============================================================================
# 2. Exact Model-Eligible Membership & Metadata Isolation Tests
# ============================================================================

class TestTask4EligibleMembershipAndMetadataIsolation(unittest.TestCase):
    """Test exact model-eligible candidate ID verification and isolated metadata streaming."""

    def test_exact_eligible_membership_success(self):
        pred_ids = ["c1", "c2", "c3"]
        expected_ids = {"c1", "c2", "c3"}
        dl_task4_utils.validate_exact_eligible_membership(pred_ids, expected_ids, "validation", "AML")

    def test_exact_eligible_membership_detects_missing_id(self):
        pred_ids = ["c1", "c2"]
        expected_ids = {"c1", "c2", "c3"}
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_exact_eligible_membership(pred_ids, expected_ids, "validation", "AML")
        self.assertIn("Missing 1 eligible IDs", str(ctx.exception))

    def test_exact_eligible_membership_detects_ineligible_extra_id(self):
        pred_ids = ["c1", "c2", "c3", "c4_ineligible"]
        expected_ids = {"c1", "c2", "c3"}
        with self.assertRaises(ValueError) as ctx:
            dl_task4_utils.validate_exact_eligible_membership(pred_ids, expected_ids, "validation", "DL")
        self.assertIn("Found 1 ineligible/extra IDs", str(ctx.exception))

    def test_load_authoritative_manifest_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            dl_task4_utils.load_authoritative_eligible_candidate_ids("nonexistent_manifest.csv", "validation", 1884)
        self.assertIn("Authoritative Task 1 prepared validation split file not found", str(ctx.exception))

    def test_load_split_candidate_metadata_retains_only_requested_split(self):
        import tempfile
        df_mixed = pd.DataFrame({
            "candidate_id": ["val_1", "test_1", "train_1"],
            "split": ["validation", "test", "train"],
            "label": [1, 0, 1],
            "between_text": ["text1", "text2", "text3"],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            df_mixed.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            df_val_meta = dl_task4_utils.load_split_candidate_metadata(tmp_path, "validation", chunksize=2)
            self.assertEqual(len(df_val_meta), 1)
            self.assertEqual(df_val_meta["candidate_id"].iloc[0], "val_1")
            self.assertEqual(df_val_meta["split"].iloc[0], "validation")
            # Verify test candidate records/labels were NOT retained in the returned dataframe
            self.assertFalse((df_val_meta["split"] == "test").any())
            self.assertNotIn("test_1", df_val_meta["candidate_id"].values)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ============================================================================
# 3. Lexicographic Validation Selection & Tie-Breaking Tests
# ============================================================================

class TestTask4LexicographicValidationSelection(unittest.TestCase):
    """Test lexicographic hyperparameter optimization and deterministic tie-breaking."""

    def test_higher_positive_f1_wins(self):
        self.assertTrue(dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.65, cand_macro_f1=0.70, cand_alpha=0.5, cand_tau=0.5,
            best_f1_pos=0.60, best_macro_f1=0.80, best_alpha=0.3, best_tau=0.4
        ))
        self.assertFalse(dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.55, cand_macro_f1=0.90, cand_alpha=0.5, cand_tau=0.5,
            best_f1_pos=0.60, best_macro_f1=0.70, best_alpha=0.3, best_tau=0.4
        ))

    def test_equal_positive_f1_higher_macro_f1_wins(self):
        # Equal pos F1 within tolerance
        self.assertTrue(dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.600001, cand_macro_f1=0.78, cand_alpha=0.5, cand_tau=0.5,
            best_f1_pos=0.600000, best_macro_f1=0.72, best_alpha=0.3, best_tau=0.4
        ))
        self.assertFalse(dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.600000, cand_macro_f1=0.70, cand_alpha=0.5, cand_tau=0.5,
            best_f1_pos=0.600001, best_macro_f1=0.75, best_alpha=0.3, best_tau=0.4
        ))

    def test_equal_positive_and_macro_f1_invokes_deterministic_tie_break(self):
        # Equal pos F1 and equal macro F1 -> prefer smaller alpha
        self.assertTrue(dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.60, cand_macro_f1=0.75, cand_alpha=0.30, cand_tau=0.50,
            best_f1_pos=0.60, best_macro_f1=0.75, best_alpha=0.60, best_tau=0.50
        ))
        # Equal pos F1, macro F1, and alpha -> prefer smaller tau
        self.assertTrue(dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.60, cand_macro_f1=0.75, cand_alpha=0.50, cand_tau=0.30,
            best_f1_pos=0.60, best_macro_f1=0.75, best_alpha=0.50, best_tau=0.60
        ))
        # Inferior candidate under tie-break
        self.assertFalse(dl_task4_utils.is_better_configuration(
            cand_f1_pos=0.60, cand_macro_f1=0.75, cand_alpha=0.70, cand_tau=0.50,
            best_f1_pos=0.60, best_macro_f1=0.75, best_alpha=0.50, best_tau=0.50
        ))

    def test_selection_is_order_invariant(self):
        val_y = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
        p_aml = np.array([0.1, 0.7, 0.2, 0.8, 0.3, 0.4, 0.1, 0.2, 0.9, 0.1])
        p_dl = np.array([0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.2, 0.1, 0.7, 0.2])

        alpha_grid = [0.2, 0.5, 0.8]
        tau_grid = [0.3, 0.5, 0.7]

        a1, t1, f1, m1, records = dl_task4_utils.select_validation_fusion_hyperparameters(
            val_y, p_aml, p_dl, alpha_grid=alpha_grid, tau_grid=tau_grid
        )

        # Run reverse grid
        a2, t2, f2, m2, _ = dl_task4_utils.select_validation_fusion_hyperparameters(
            val_y, p_aml, p_dl, alpha_grid=list(reversed(alpha_grid)), tau_grid=list(reversed(tau_grid))
        )

        self.assertEqual(a1, a2)
        self.assertEqual(t1, t2)
        self.assertAlmostEqual(f1, f2, places=6)
        self.assertAlmostEqual(m1, m2, places=6)


# ============================================================================
# 4. Linguistic Sanity Checks & Diagnostics Tests
# ============================================================================

class TestTask4LinguisticSanityChecks(unittest.TestCase):
    """Test explicit negation cue detection, distance window, and non-overlapping diagnostics."""

    def test_negation_pattern_detects_cues(self):
        samples = [
            "does not activate transcription",
            "failed to induce phosphorylation",
            "showed no significant effect",
            "without PmrA expression",
            "neither kinase nor regulator",
        ]
        for s in samples:
            self.assertIsNotNone(NEGATION_PATTERN.search(s), f"Failed to match negation in: {s}")

    def test_linguistic_filtering_non_overlapping_diagnostics(self):
        df_test = pd.DataFrame({
            "candidate_id": ["c1", "c2", "c3", "c4", "c5"],
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


# ============================================================================
# 5. Prediction and Relation Export Verification Tests
# ============================================================================

class TestTask4ExportVerification(unittest.TestCase):
    """Test read-back verification of prediction and relation artifacts."""

    def test_prediction_export_verification_success(self):
        import tempfile
        frozen_alpha = 0.60
        frozen_tau = 0.50

        p_dl = np.array([0.8, 0.2])
        p_aml = np.array([0.6, 0.4])
        p_hyb = frozen_alpha * p_dl + (1.0 - frozen_alpha) * p_aml

        df_exp = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "true_label": [1, 0],
            "aml_prob": p_aml,
            "dl_prob": p_dl,
            "hybrid_prob": p_hyb,
            "aml_pred": [1, 0],
            "dl_pred": [1, 0],
            "hybrid_pred": [1, 0],
            "split": ["validation", "validation"],
        })

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            df_exp.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            dl_task4_utils.verify_prediction_export(
                filepath=tmp_path,
                expected_rows=2,
                split_name="validation",
                source_candidate_ids=["c1", "c2"],
                source_true_labels=[1, 0],
                source_aml_probs=p_aml,
                source_dl_probs=p_dl,
                frozen_alpha=frozen_alpha,
                frozen_tau=frozen_tau,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_prediction_export_verification_detects_formula_mismatch(self):
        import tempfile
        df_bad = pd.DataFrame({
            "candidate_id": ["c1"],
            "true_label": [1],
            "aml_prob": [0.5],
            "dl_prob": [0.5],
            "hybrid_prob": [0.99],  # Wrong! Should be 0.50
            "aml_pred": [1],
            "dl_pred": [1],
            "hybrid_pred": [1],
            "split": ["validation"],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            df_bad.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_prediction_export(
                    filepath=tmp_path,
                    expected_rows=1,
                    split_name="validation",
                    source_candidate_ids=["c1"],
                    source_true_labels=[1],
                    source_aml_probs=[0.5],
                    source_dl_probs=[0.5],
                    frozen_alpha=0.5,
                    frozen_tau=0.5,
                )
            self.assertIn("Exported hybrid_prob mismatch with recomputed formula", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_relation_export_verification_success(self):
        import tempfile
        rec = dl_utils.build_unified_relation_record(
            candidate_id="PMC123_T1_T2",
            article_id="PMC123",
            section_id="sec_01",
            cause_id="T1",
            cause_text="sensor",
            cause_start=10,
            cause_end=16,
            effect_id="T2",
            effect_text="regulator",
            effect_start=30,
            effect_end=39,
            confidence=0.88,
            extraction_method="hybrid_probability_fusion",
            evidence_text="sensor activates regulator",
            split="test",
            direction="cause->effect",
            provenance="PubMedBERT_alpha0.60+LogisticRegression_1to2",
        )
        df_rels = pd.DataFrame([rec])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            df_rels.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            dl_task4_utils.verify_relation_export(
                filepath=tmp_path,
                expected_count=1,
                sanitized_candidate_ids=["PMC123_T1_T2"],
                candidate_id_to_relation_id={"PMC123_T1_T2": rec["relation_id"]},
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_relation_export_verification_detects_self_referential_relation(self):
        import tempfile
        rec = dl_utils.build_unified_relation_record(
            candidate_id="PMC123_T1_T1",
            article_id="PMC123",
            section_id="sec_01",
            cause_id="T1",
            cause_text="sensor",
            cause_start=10,
            cause_end=16,
            effect_id="T1",  # Self-referential!
            effect_text="sensor",
            effect_start=10,
            effect_end=16,
            confidence=0.88,
            extraction_method="hybrid_probability_fusion",
            evidence_text="sensor activates sensor",
            split="test",
            direction="cause->effect",
            provenance="PubMedBERT_alpha0.60+LogisticRegression_1to2",
        )
        df_rels = pd.DataFrame([rec])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            df_rels.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                dl_task4_utils.verify_relation_export(filepath=tmp_path, expected_count=1)
            self.assertIn("self-referential relations", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ============================================================================
# 6. Graph Pruning & Bayesian Network Structural Compilation Tests
# ============================================================================

class TestTask4DAGAndBayesianNetwork(unittest.TestCase):
    """Test DAG cycle resolution and DiscreteBayesianNetwork structural compilation."""

    def test_enforce_dag_acyclicity_removes_self_loops(self):
        G = nx.DiGraph()
        G.add_edge("A", "A", confidence=0.9)
        G.add_edge("A", "B", confidence=0.8)

        G_dag, audit_log = enforce_dag_acyclicity(G, weight_attr="confidence")
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertFalse(G_dag.has_edge("A", "A"))
        self.assertTrue(G_dag.has_edge("A", "B"))
        self.assertTrue(any(e.get("reason") == "self_loop" for e in audit_log))

    def test_enforce_dag_acyclicity_resolves_bidirectional_conflicts(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", confidence=0.85)
        G.add_edge("B", "A", confidence=0.45)

        G_dag, audit_log = enforce_dag_acyclicity(G, weight_attr="confidence")
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertTrue(G_dag.has_edge("A", "B"))
        self.assertFalse(G_dag.has_edge("B", "A"))
        self.assertTrue(any("bidirectional" in e.get("reason", "") for e in audit_log))

    def test_enforce_dag_acyclicity_removes_feedback_cycles(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", confidence=0.90)
        G.add_edge("B", "C", confidence=0.80)
        G.add_edge("C", "A", confidence=0.60)

        self.assertFalse(nx.is_directed_acyclic_graph(G))
        G_dag, audit_log = enforce_dag_acyclicity(G, weight_attr="confidence")
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertTrue(G_dag.has_edge("A", "B"))
        self.assertTrue(G_dag.has_edge("B", "C"))
        self.assertFalse(G_dag.has_edge("C", "A"))
        self.assertTrue(any("cycle" in e.get("reason", "") for e in audit_log))

    def test_compile_discrete_bayesian_network_preserves_all_nodes_and_edges(self):
        """Bayesian network compilation must preserve all nodes (even isolated ones) and edges."""
        G = nx.DiGraph()
        G.add_node("Isolated_Node")
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        bn = dl_task4_utils.compile_discrete_bayesian_network(G)
        self.assertEqual(set(bn.nodes()), set(G.nodes()))
        self.assertEqual(set(bn.edges()), set(G.edges()))
        self.assertIn("Isolated_Node", bn.nodes())
        self.assertTrue(nx.is_directed_acyclic_graph(bn))

    def test_pgmpy_illustrative_bayesian_network_do_calculus(self):
        """Illustrative regulatory cascade with synthetic CPDs demonstrates observational vs interventional query."""
        bn = DiscreteBayesianNetwork([("S", "R"), ("R", "T")])

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

        # Interventional query: P(T=1 | do(R=1))
        interv_bn = bn.do(["R"])
        infer_interv = VariableElimination(interv_bn)
        q_interv = infer_interv.query(["T"], evidence={"R": "1"})
        p_interv = float(q_interv.values[1])

        self.assertGreater(p_obs, 0.0)
        self.assertGreater(p_interv, 0.0)
        self.assertNotEqual(p_obs, p_interv)


if __name__ == "__main__":
    unittest.main()
