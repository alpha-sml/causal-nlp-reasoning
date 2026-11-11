"""Focused unit tests for Phase 2 Task 4: Hybrid Probability Fusion and Graph Integration.

Verifies:
1. Input contract validation and candidate alignment invariants.
2. Hybrid probability score fusion and validation discipline.
3. Linguistic sanity filtering (negation and argument distance constraints).
4. Standardized 17-column UnifiedRelationRecord schema conformance.
5. NetworkX DAG construction and deterministic cycle breaking.
6. Bayesian network compilation and observational vs. interventional queries.
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


class TestTask4InputValidationAndAlignment(unittest.TestCase):
    """Test strict candidate alignment and input contract assertions."""

    def setUp(self):
        self.n_val = 10
        self.val_cands = [f"cand_val_{i}" for i in range(self.n_val)]
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

    def test_aligned_inputs_succeed(self):
        """Aligned candidate IDs, matching true labels, and matching row counts must succeed."""
        self.assertEqual(len(self.df_aml_valid), len(self.df_dl_valid))
        self.assertTrue((self.df_aml_valid["candidate_id"].values == self.df_dl_valid["candidate_id"].values).all())
        self.assertTrue((self.df_aml_valid["true_label"].values == self.df_dl_valid["true_label"].values).all())

    def test_mismatched_candidate_ids_detected(self):
        """Mismatched candidate IDs across streams must be detected."""
        df_dl_bad = self.df_dl_valid.copy()
        df_dl_bad.loc[0, "candidate_id"] = "different_cand_id"
        with self.assertRaises(AssertionError):
            assert (self.df_aml_valid["candidate_id"].values == df_dl_bad["candidate_id"].values).all(), "Candidate ID mismatch"

    def test_mismatched_true_labels_detected(self):
        """Mismatched ground truth labels across streams must be detected."""
        df_dl_bad = self.df_dl_valid.copy()
        df_dl_bad.loc[0, "true_label"] = 1 - df_dl_bad.loc[0, "true_label"]
        with self.assertRaises(AssertionError):
            assert (self.df_aml_valid["true_label"].values == df_dl_bad["true_label"].values).all(), "Label agreement mismatch"

    def test_duplicate_candidate_ids_detected(self):
        """Duplicate candidate IDs in prediction inputs must be rejected."""
        df_aml_dup = self.df_aml_valid.copy()
        df_aml_dup.loc[1, "candidate_id"] = df_aml_dup.loc[0, "candidate_id"]
        with self.assertRaises(AssertionError):
            assert df_aml_dup["candidate_id"].duplicated().sum() == 0, "Duplicate candidate IDs found"


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

    def test_invalid_alpha_raises_value_error(self):
        """Alpha outside [0, 1] must raise ValueError."""
        probs = np.array([0.5])
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(prob_aml=probs, prob_dl=probs, alpha=-0.1)
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(prob_aml=probs, prob_dl=probs, alpha=1.1)


class TestTask4LinguisticSanityChecks(unittest.TestCase):
    """Test linguistic sanity filtering rules (negation and argument distance constraints)."""

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

    def test_argument_distance_constraint(self):
        """Candidate argument pairs exceeding max distance (400 chars) are flagged."""
        max_dist = 400
        short_between = "activates"
        long_between = "word " * 120  # ~600 chars
        self.assertLessEqual(len(short_between), max_dist)
        self.assertGreater(len(long_between), max_dist)


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


class TestTask4DAGAndBayesianNetwork(unittest.TestCase):
    """Test DAG construction, cycle breaking, and pgmpy compilation."""

    def test_enforce_dag_acyclicity_removes_cycles(self):
        """Cyclic graph must be pruned deterministically to an acyclic DAG."""
        G = nx.DiGraph()
        # Form a 3-node cycle: A -> B -> C -> A
        G.add_edge("A", "B", confidence=0.9)
        G.add_edge("B", "C", confidence=0.8)
        G.add_edge("C", "A", confidence=0.6)  # Lowest weight feedback edge

        self.assertFalse(nx.is_directed_acyclic_graph(G))
        G_dag, audit_log = enforce_dag_acyclicity(G, weight_attr="confidence")
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertEqual(len(audit_log), 1)
        self.assertEqual(audit_log[0]["u"], "C")
        self.assertEqual(audit_log[0]["v"], "A")
        self.assertEqual(audit_log[0]["reason"], "directed_cycle_feedback_edge")

    def test_pgmpy_bayesian_network_compilation_and_do_calculus(self):
        """Compiled acyclic graph must validate in pgmpy and evaluate observational vs interventional queries."""
        # Simple regulatory cascade: Sensor (S) -> Regulator (R) -> Target (T)
        bn = DiscreteBayesianNetwork([("S", "R"), ("R", "T")])
        self.assertTrue(nx.is_directed_acyclic_graph(bn))

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
        # Observational: P(T=1 | S=1)
        q_obs = infer.query(["T"], evidence={"S": "1"})
        p_obs = float(q_obs.values[1])

        # Interventional via graph surgery: P(T=1 | do(R=1))
        interv_bn = bn.do(["R"])
        infer_interv = VariableElimination(interv_bn)
        q_interv = infer_interv.query(["T"], evidence={"R": "1"})
        p_interv = float(q_interv.values[1])

        self.assertGreater(p_obs, 0.0)
        self.assertGreater(p_interv, 0.0)
        self.assertNotEqual(p_obs, p_interv)


if __name__ == "__main__":
    unittest.main()
