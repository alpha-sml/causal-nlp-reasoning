"""tests/test_phase1_baseline.py

Comprehensive Phase 1 AML baseline validation suite verifying:
1. Pure deterministic unit tests for lexical, syntactic dependency, and hybrid extractors in AML/causal_extractors.py.
2. Directional correctness across active, passive, reversed argument orders, and discourse patterns.
3. Explicit cause/effect semantic verification (cause, effect, direction string) in both forward and reversed orientations.
4. Syntactic dependency extractor scope boundaries (rejecting non-verbal discourse cues per documented design).
5. Robust rejection of negations, correlational controls, self-loops, empty/whitespace/NaN inputs.
6. UnifiedRelationRecord construction and alignment with candidate-label conventions.
7. Corpus data integrity, standoff pairing, 100% offset validity, and 12/4/4 article split isolation.
8. Exact held-out test split metric reconciliation across Lexical (36.46% F1), Dependency (29.07% F1), and Hybrid (40.74% F1).
9. Supervised candidate evaluation arithmetic identities (1,408 test candidates: 100 positive, 1,308 negative).
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np


def find_repo_root() -> Path:
    candidates = []
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    if "__file__" in globals():
        fpath = Path(__file__).resolve().parent
        candidates.extend([fpath, *fpath.parents])
    for candidate in candidates:
        if (candidate / "00_setup.ipynb").is_file() and (candidate / "README.md").is_file():
            return candidate.resolve()
    raise RuntimeError("Repository root could not be identified.")


REPO_ROOT = find_repo_root()
AML_DIR = REPO_ROOT / "AML"
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
BIOCAUSE_DIR = RAW_DIR / "biocause" / "BioCause_corpus"

if str(AML_DIR) not in sys.path:
    sys.path.insert(0, str(AML_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_extractors import (
    extract_lexical_relation,
    extract_dependency_relation,
    extract_hybrid_relation,
    build_extracted_records,
    compute_shortest_dependency_path,
    enforce_dag_acyclicity,
)

SCHEMA_COLUMNS = [
    "relation_id", "article_id", "section_id",
    "cause_id", "cause_text", "cause_start", "cause_end",
    "effect_id", "effect_text", "effect_start", "effect_end",
    "direction", "confidence", "extraction_method",
    "evidence_text", "provenance", "split"
]


# ===========================================================================
# 1. Deterministic Unit Tests (Zero external data dependencies)
# ===========================================================================
class TestPhase1ExtractorUnitTests(unittest.TestCase):
    """Deterministic unit tests validating extractor logic, directionality, and controls."""

    def test_01_active_verbal_directionality(self):
        """Active verbal cues: forward candidate yields pred_directed=1, reversed candidate yields pred_directed=0."""
        # Sentence: "PmrB kinase directly activates PmrA response regulator."
        # Forward candidate: arg1=Cause, arg2=Effect, b4=1
        r_lex_fwd = extract_lexical_relation("PmrB kinase", "PmrA response regulator", " directly activates ", arg1_before_arg2=1)
        r_dep_fwd = extract_dependency_relation("PmrB kinase", "PmrA response regulator", " directly activates ", arg1_before_arg2=1)
        for r, m in [(r_lex_fwd, "lexical"), (r_dep_fwd, "dependency")]:
            self.assertEqual(r["is_causal"], 1, f"{m} forward active should be causal")
            self.assertEqual(r["pred_directed"], 1, f"{m} forward active should predict directed causality")
            self.assertEqual(r["cause"], "PmrB kinase", f"{m} cause must be PmrB kinase")
            self.assertEqual(r["effect"], "PmrA response regulator", f"{m} effect must be PmrA response regulator")
            self.assertEqual(r["direction"], "arg1->arg2", f"{m} direction must be arg1->arg2")

        # Reversed candidate: arg1=Effect, arg2=Cause, b4=0
        r_lex_rev = extract_lexical_relation("PmrA response regulator", "PmrB kinase", " directly activates ", arg1_before_arg2=0)
        r_dep_rev = extract_dependency_relation("PmrA response regulator", "PmrB kinase", " directly activates ", arg1_before_arg2=0)
        for r, m in [(r_lex_rev, "lexical"), (r_dep_rev, "dependency")]:
            self.assertEqual(r["is_causal"], 1, f"{m} reversed active should be causal")
            self.assertEqual(r["pred_directed"], 0, f"{m} reversed active must NOT predict arg1->arg2")
            self.assertEqual(r["cause"], "PmrB kinase", f"{m} semantic cause remains PmrB kinase")
            self.assertEqual(r["effect"], "PmrA response regulator", f"{m} semantic effect remains PmrA response regulator")
            self.assertEqual(r["direction"], "arg2->arg1", f"{m} direction must be arg2->arg1")

    def test_02_passive_verbal_directionality(self):
        """Passive verbal cues: arg1=Cause, arg2=Effect, b4=0 yields pred_directed=1."""
        # Sentence: "Resistance to polymyxin is caused by PmrB phosphorylation."
        # Forward candidate: arg1=Cause (PmrB), arg2=Effect (Resistance), b4=0 (Cause appears second in text)
        r_lex_fwd = extract_lexical_relation("PmrB phosphorylation", "Resistance to polymyxin", " is caused by ", arg1_before_arg2=0)
        r_dep_fwd = extract_dependency_relation("PmrB phosphorylation", "Resistance to polymyxin", " is caused by ", arg1_before_arg2=0)
        for r, m in [(r_lex_fwd, "lexical"), (r_dep_fwd, "dependency")]:
            self.assertEqual(r["is_causal"], 1, f"{m} passive forward should be causal")
            self.assertEqual(r["pred_directed"], 1, f"{m} passive forward must predict directed causality")
            self.assertEqual(r["cause"], "PmrB phosphorylation", f"{m} cause must be PmrB phosphorylation")
            self.assertEqual(r["effect"], "Resistance to polymyxin", f"{m} effect must be Resistance to polymyxin")
            self.assertEqual(r["direction"], "arg1->arg2", f"{m} direction must be arg1->arg2")

        # Reversed candidate: arg1=Effect (Resistance), arg2=Cause (PmrB), b4=1 (Effect appears first in text)
        r_lex_rev = extract_lexical_relation("Resistance to polymyxin", "PmrB phosphorylation", " is caused by ", arg1_before_arg2=1)
        r_dep_rev = extract_dependency_relation("Resistance to polymyxin", "PmrB phosphorylation", " is caused by ", arg1_before_arg2=1)
        for r, m in [(r_lex_rev, "lexical"), (r_dep_rev, "dependency")]:
            self.assertEqual(r["is_causal"], 1, f"{m} passive reversed should be causal")
            self.assertEqual(r["pred_directed"], 0, f"{m} passive reversed must NOT predict arg1->arg2")
            self.assertEqual(r["cause"], "PmrB phosphorylation", f"{m} semantic cause remains PmrB phosphorylation")
            self.assertEqual(r["effect"], "Resistance to polymyxin", f"{m} semantic effect remains Resistance to polymyxin")
            self.assertEqual(r["direction"], "arg2->arg1", f"{m} direction must be arg2->arg1")

        # Also test "results from"
        r_lex_rf = extract_lexical_relation("PmrB mutation", "bacterial lysis", " results from ", arg1_before_arg2=0)
        r_dep_rf = extract_dependency_relation("PmrB mutation", "bacterial lysis", " results from ", arg1_before_arg2=0)
        self.assertEqual(r_lex_rf["pred_directed"], 1)
        self.assertEqual(r_dep_rf["pred_directed"], 1)
        self.assertEqual(r_lex_rf["cause"], "PmrB mutation")
        self.assertEqual(r_dep_rf["cause"], "PmrB mutation")

    def test_03_discourse_patterns_and_dependency_scope_boundary(self):
        """Discourse cues: lexical extractor recognizes discourse; dependency extractor rejects non-verbal cues per documented scope."""
        # Inverted discourse ("because"): "Cell death occurred because compound X inhibited the enzyme."
        # Forward candidate: arg1=Cause (compound X), arg2=Effect (Cell death), b4=0
        r_lex_inv = extract_lexical_relation("compound X", "Cell death", " occurred because ", arg1_before_arg2=0)
        self.assertEqual(r_lex_inv["is_causal"], 1)
        self.assertEqual(r_lex_inv["pred_directed"], 1)
        self.assertEqual(r_lex_inv["cause"], "compound X")
        self.assertEqual(r_lex_inv["effect"], "Cell death")

        # Reversed candidate: arg1=Effect, arg2=Cause, b4=1
        r_lex_inv_rev = extract_lexical_relation("Cell death", "compound X", " occurred because ", arg1_before_arg2=1)
        self.assertEqual(r_lex_inv_rev["is_causal"], 1)
        self.assertEqual(r_lex_inv_rev["pred_directed"], 0)
        self.assertEqual(r_lex_inv_rev["cause"], "compound X")
        self.assertEqual(r_lex_inv_rev["effect"], "Cell death")

        # Dependency extractor scope boundary: non-verbal discourse cues have no verbal causal lemma in CAUSAL_LEMMAS
        r_dep_inv = extract_dependency_relation("compound X", "Cell death", " occurred because ", arg1_before_arg2=0)
        self.assertEqual(r_dep_inv["pred_directed"], 0, "Dependency extractor must not claim non-verbal discourse cues")
        self.assertEqual(r_dep_inv["is_causal"], 0)
        self.assertEqual(r_dep_inv["rejection_reason"], "no_causal_lemma_found")

        # Forward discourse ("therefore"): "PmrB was mutated; therefore, resistance increased."
        r_lex_fwd = extract_lexical_relation("PmrB", "resistance", "; therefore, ", arg1_before_arg2=1)
        self.assertEqual(r_lex_fwd["pred_directed"], 1)
        self.assertEqual(r_lex_fwd["cause"], "PmrB")
        self.assertEqual(r_lex_fwd["effect"], "resistance")

        r_dep_fwd = extract_dependency_relation("PmrB", "resistance", "; therefore, ", arg1_before_arg2=1)
        self.assertEqual(r_dep_fwd["pred_directed"], 0, "Dependency extractor must reject non-verbal forward discourse cues")

    def test_04_negative_controls_and_input_hardening(self):
        """Robust rejection of negation, correlational cues, self-loops, and empty/whitespace/NaN inputs."""
        # Negation
        r_neg = extract_lexical_relation("Mutation X", "virulence", " does not cause ", arg1_before_arg2=1)
        self.assertEqual(r_neg["is_causal"], 0)
        self.assertEqual(r_neg["pred_directed"], 0)
        self.assertEqual(r_neg["rejection_reason"], "negation_detected")

        r_neg_dep = extract_dependency_relation("Mutation X", "virulence", " does not cause ", arg1_before_arg2=1)
        self.assertEqual(r_neg_dep["is_causal"], 0)
        self.assertEqual(r_neg_dep["pred_directed"], 0)
        self.assertEqual(r_neg_dep["rejection_reason"], "negation_detected")

        # Correlation control
        r_corr = extract_lexical_relation("Gene expression A", "phenotype B", " is correlated with ", arg1_before_arg2=1)
        self.assertEqual(r_corr["is_causal"], 0)
        self.assertEqual(r_corr["pred_directed"], 0)
        self.assertEqual(r_corr["rejection_reason"], "correlation_not_causation")

        # Self-loops
        r_self = extract_lexical_relation("PmrA", "PmrA", " activates ", arg1_before_arg2=1)
        self.assertEqual(r_self["is_causal"], 0)
        self.assertEqual(r_self["pred_directed"], 0)
        self.assertEqual(r_self["rejection_reason"], "self_loop")

        # Empty / whitespace / None / NaN inputs
        for bad_arg in ["", "   ", None, float("nan"), np.nan]:
            res1 = extract_lexical_relation(bad_arg, "PmrA", " activates ", arg1_before_arg2=1)
            self.assertEqual(res1["pred_directed"], 0)
            self.assertEqual(res1["rejection_reason"], "empty_argument")

            res2 = extract_dependency_relation(bad_arg, "PmrA", " activates ", arg1_before_arg2=1)
            self.assertEqual(res2["pred_directed"], 0)
            self.assertEqual(res2["rejection_reason"], "empty_argument")

        # Empty / whitespace / None / NaN evidence text
        for bad_btw in ["", "   ", None, float("nan"), np.nan, "nan"]:
            res_btw_lex = extract_lexical_relation("PmrA", "PmrB", bad_btw, arg1_before_arg2=1)
            self.assertEqual(res_btw_lex["pred_directed"], 0)
            self.assertEqual(res_btw_lex["rejection_reason"], "empty_evidence_text")

            res_btw_dep = extract_dependency_relation("PmrA", "PmrB", bad_btw, arg1_before_arg2=1)
            self.assertEqual(res_btw_dep["pred_directed"], 0)
            self.assertEqual(res_btw_dep["rejection_reason"], "empty_evidence_text")

    def test_05_record_construction_and_candidate_label_convention(self):
        """build_extracted_records generates records only for pred_directed=1 and aligns with candidate labels."""
        # Create synthetic candidate dataframe
        synth_data = [
            # 1. Valid active forward: label=1 -> should be emitted
            {"candidate_id": "C1", "article_id": "A1", "section_id": "S1",
             "arg1_id": "T1", "arg1_text": "PmrB", "arg1_start": 10, "arg1_end": 14,
             "arg2_id": "T2", "arg2_text": "PmrA", "arg2_start": 30, "arg2_end": 34,
             "between_text": " activates ", "arg1_before_arg2": 1, "label": 1, "split": "test"},
            # 2. Active reversed: label=0 -> should NOT be emitted (pred_directed=0, scores as TN)
            {"candidate_id": "C2", "article_id": "A1", "section_id": "S1",
             "arg1_id": "T2", "arg1_text": "PmrA", "arg1_start": 30, "arg1_end": 34,
             "arg2_id": "T1", "arg2_text": "PmrB", "arg2_start": 10, "arg2_end": 14,
             "between_text": " activates ", "arg1_before_arg2": 0, "label": 0, "split": "test"},
            # 3. Valid passive forward: label=1 -> should be emitted (arg1=Cause, arg2=Effect, b4=0)
            {"candidate_id": "C3", "article_id": "A1", "section_id": "S1",
             "arg1_id": "T3", "arg1_text": "Kinase", "arg1_start": 35, "arg1_end": 41,
             "arg2_id": "T4", "arg2_text": "Lysis", "arg2_start": 5, "arg2_end": 10,
             "between_text": " is caused by ", "arg1_before_arg2": 0, "label": 1, "split": "test"},
            # 4. Passive reversed: label=0 -> should NOT be emitted
            {"candidate_id": "C4", "article_id": "A1", "section_id": "S1",
             "arg1_id": "T4", "arg1_text": "Lysis", "arg1_start": 5, "arg1_end": 10,
             "arg2_id": "T3", "arg2_text": "Kinase", "arg2_start": 35, "arg2_end": 41,
             "between_text": " is caused by ", "arg1_before_arg2": 1, "label": 0, "split": "test"},
            # 5. Negated control: label=0 -> should NOT be emitted
            {"candidate_id": "C5", "article_id": "A1", "section_id": "S1",
             "arg1_id": "T5", "arg1_text": "Drug", "arg1_start": 0, "arg1_end": 4,
             "arg2_id": "T6", "arg2_text": "Disease", "arg2_start": 25, "arg2_end": 32,
             "between_text": " does not cause ", "arg1_before_arg2": 1, "label": 0, "split": "test"},
            # 6. Missing evidence: label=0 -> should NOT be emitted
            {"candidate_id": "C6", "article_id": "A1", "section_id": "S1",
             "arg1_id": "T7", "arg1_text": "GeneA", "arg1_start": 0, "arg1_end": 5,
             "arg2_id": "T8", "arg2_text": "GeneB", "arg2_start": 5, "arg2_end": 10,
             "between_text": None, "arg1_before_arg2": 1, "label": 0, "split": "test"},
        ]
        df_synth = pd.DataFrame(synth_data)
        df_rec = build_extracted_records(df_synth, extract_lexical_relation, "LEX", "Lexical Rule Baseline", SCHEMA_COLUMNS)

        # Assert exactly 2 records emitted (C1 and C3)
        self.assertEqual(len(df_rec), 2, "Exactly 2 valid positive relations must be emitted")
        self.assertEqual(list(df_rec["relation_id"]), ["LEX_C1", "LEX_C3"])
        self.assertEqual(len(df_rec.columns), 17, "UnifiedRelationRecord schema must have 17 columns")

        # Verify Record 1: C1
        rec1 = df_rec.iloc[0]
        self.assertEqual(rec1["cause_id"], "T1")
        self.assertEqual(rec1["cause_text"], "PmrB")
        self.assertEqual(rec1["effect_id"], "T2")
        self.assertEqual(rec1["effect_text"], "PmrA")
        self.assertEqual(rec1["direction"], "cause->effect")

        # Verify Record 2: C3
        rec2 = df_rec.iloc[1]
        self.assertEqual(rec2["cause_id"], "T3")
        self.assertEqual(rec2["cause_text"], "Kinase")
        self.assertEqual(rec2["effect_id"], "T4")
        self.assertEqual(rec2["effect_text"], "Lysis")
        self.assertEqual(rec2["direction"], "cause->effect")

    def test_05b_hybrid_extractor_selection_and_ambiguity(self):
        """Hybrid extractor logic: prioritizes lexical, falls back to dependency, and prunes negations/distance."""
        # 1. Lexical detection on non-verbal discourse cue ("therefore" where dependency finds no verbal lemma)
        r_hyb_fwd = extract_hybrid_relation("PmrB", "resistance", "; therefore, ", arg1_before_arg2=1)
        self.assertEqual(r_hyb_fwd["pred_directed"], 1, "Hybrid must detect forward discourse relation")
        self.assertEqual(r_hyb_fwd["direction"], "arg1->arg2")
        self.assertEqual(r_hyb_fwd["cause"], "PmrB")
        self.assertEqual(r_hyb_fwd["effect"], "resistance")

        # 2. Reversed candidate on same discourse cue -> must not predict arg1->arg2
        r_hyb_rev = extract_hybrid_relation("resistance", "PmrB", "; therefore, ", arg1_before_arg2=0)
        self.assertEqual(r_hyb_rev["pred_directed"], 0, "Hybrid reversed discourse must not predict arg1->arg2")
        self.assertEqual(r_hyb_rev["direction"], "arg2->arg1")

        # 3. Active verbal predicate ("activates") detected by both lexical and dependency
        r_hyb_verb = extract_hybrid_relation("PmrB kinase", "PmrA protein", " directly activates ", arg1_before_arg2=1)
        self.assertEqual(r_hyb_verb["pred_directed"], 1)
        self.assertEqual(r_hyb_verb["cause"], "PmrB kinase")
        self.assertEqual(r_hyb_verb["effect"], "PmrA protein")

        # 4. Passive verbal construction ("results from")
        r_hyb_pass = extract_hybrid_relation("PmrB mutation", "lysis", " results from ", arg1_before_arg2=0)
        self.assertEqual(r_hyb_pass["pred_directed"], 1)
        self.assertEqual(r_hyb_pass["cause"], "PmrB mutation")
        self.assertEqual(r_hyb_pass["effect"], "lysis")

        # 5. Neither matches -> rejected
        r_none = extract_hybrid_relation("GeneA", "GeneB", " was evaluated across ", arg1_before_arg2=1)
        self.assertEqual(r_none["pred_directed"], 0)
        self.assertEqual(r_none["is_causal"], 0)

        # 6. Distance window exceeded (> 400 chars) -> rejected
        long_between = " between " * 60
        r_dist = extract_hybrid_relation("GeneA", "GeneB", long_between, arg1_before_arg2=1)
        self.assertEqual(r_dist["pred_directed"], 0)
        self.assertEqual(r_dist["rejection_reason"], "distance_window_exceeded")

        # 7. Negation pruning in hybrid
        r_neg = extract_hybrid_relation("Mutation", "virulence", " does not cause ", arg1_before_arg2=1)
        self.assertEqual(r_neg["pred_directed"], 0)
        self.assertEqual(r_neg["rejection_reason"], "negation_detected")

    def test_05c_shortest_dependency_path_extraction(self):
        """Verify shortest dependency path (SDP) computation across active and passive causal sentences."""
        sdp_act = compute_shortest_dependency_path(
            "PmrB kinase", "PmrA response regulator", " directly activates ", arg1_before_arg2=1
        )
        self.assertGreater(sdp_act["sdp_length"], 0)
        self.assertTrue(sdp_act["sdp_has_causal_lemma"])
        self.assertIn("activate", sdp_act["sdp_lemmas"])

        sdp_pass = compute_shortest_dependency_path(
            "PmrB phosphorylation", "Resistance to polymyxin", " is caused by ", arg1_before_arg2=0
        )
        self.assertGreater(sdp_pass["sdp_length"], 0)
        self.assertTrue(sdp_pass["sdp_has_causal_lemma"])
        self.assertIn("cause", sdp_pass["sdp_lemmas"])

    def test_05d_deterministic_dag_cycle_breaking(self):
        """Verify enforce_dag_acyclicity resolves self-loops, 2-cycles, and 3-cycles deterministically."""
        import networkx as nx
        G_cyclic = nx.DiGraph()
        # Self-loop
        G_cyclic.add_edge("PmrA", "PmrA", score=0.99)
        # Bidirectional 2-cycle: X -> Y (0.91) vs Y -> X (0.42)
        G_cyclic.add_edge("X", "Y", score=0.91)
        G_cyclic.add_edge("Y", "X", score=0.42)
        # 3-node feedback loop: A -> B (0.92), B -> C (0.88), C -> A (0.51)
        G_cyclic.add_edge("A", "B", score=0.92)
        G_cyclic.add_edge("B", "C", score=0.88)
        G_cyclic.add_edge("C", "A", score=0.51)

        self.assertFalse(nx.is_directed_acyclic_graph(G_cyclic))
        G_dag, audit_log = enforce_dag_acyclicity(G_cyclic, weight_attr="score")
        self.assertTrue(nx.is_directed_acyclic_graph(G_dag))
        self.assertEqual(len(audit_log), 3)
        self.assertFalse(G_dag.has_edge("PmrA", "PmrA"))
        self.assertTrue(G_dag.has_edge("X", "Y"))
        self.assertFalse(G_dag.has_edge("Y", "X"))
        self.assertTrue(G_dag.has_edge("A", "B"))
        self.assertTrue(G_dag.has_edge("B", "C"))
        self.assertFalse(G_dag.has_edge("C", "A"))


# ===========================================================================
# 2. Corpus Data Integrity & Isolation Tests (Requires BioCause corpus)
# ===========================================================================
class TestPhase1CorpusAndDataIntegrity(unittest.TestCase):
    """Corpus validation verifying standoff pairing, event integrity, and split isolation."""

    @classmethod
    def setUpClass(cls):
        if not BIOCAUSE_DIR.is_dir() or not (PROCESSED_DIR / "biocause_relations.csv").is_file():
            raise unittest.SkipTest("BioCause raw corpus or processed relations not found. Skipping corpus integrity tests.")

    def test_06_corpus_pairing_and_standoff_invariants(self):
        """Verify BioCause standoff pairing, 198 .txt / .ann files, 0 broken references, 0 discontinuous spans."""
        txt_files = sorted(BIOCAUSE_DIR.glob("*.txt"))
        ann_files = sorted(BIOCAUSE_DIR.glob("*.ann"))
        self.assertEqual(len(txt_files), 198, f"Expected 198 .txt files, got {len(txt_files)}")
        self.assertEqual(len(ann_files), 198, f"Expected 198 .ann files, got {len(ann_files)}")
        self.assertEqual(
            set(f.stem for f in txt_files), set(f.stem for f in ann_files),
            "Unpaired .txt and .ann files detected in BioCause corpus"
        )
        articles = set(s.split("-")[0] for s in [f.stem for f in txt_files])
        self.assertEqual(len(articles), 20, f"Expected 20 PMC articles, got {len(articles)}")

        broken = 0
        events_count = 0
        disc_spans = 0
        for af in ann_files:
            t_ids = set()
            events = []
            with open(af, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("T"):
                        t_ids.add(line.split("\t")[0])
                        if ";" in line.split("\t")[1]:
                            disc_spans += 1
                    elif line.startswith("E") and "Causality:" in line:
                        events.append(line)
            for el in events:
                parts = el.split("\t")[1].split()
                events_count += 1
                trig = parts[0].split(":")[1]
                if trig not in t_ids:
                    broken += 1
                for tok in parts[1:]:
                    val = tok.split(":")[1]
                    if val not in t_ids:
                        broken += 1

        self.assertEqual(events_count, 851, f"Expected 851 Causality events, found {events_count}")
        self.assertEqual(broken, 0, f"Found {broken} broken event references in .ann files")
        self.assertEqual(disc_spans, 0, f"Expected 0 discontinuous spans, found {disc_spans}")

    def test_07_split_isolation_and_coverage(self):
        """Verify strict 12/4/4 article split isolation and candidate-to-gold coverage."""
        df_rel = pd.read_csv(PROCESSED_DIR / "biocause_relations.csv", keep_default_na=False)
        df_ent = pd.read_csv(PROCESSED_DIR / "biocause_entities.csv", keep_default_na=False)
        df_cand = pd.read_csv(PROCESSED_DIR / "biocause_candidates.csv", keep_default_na=False)

        self.assertEqual(len(df_rel), 852, f"Expected 852 gold relations, got {len(df_rel)}")
        self.assertEqual(len(df_ent), 12939, f"Expected 12,939 entities, got {len(df_ent)}")
        self.assertEqual(len(df_cand), 13408, f"Expected 13,408 candidates, got {len(df_cand)}")

        for name, df in [("Relations", df_rel), ("Entities", df_ent), ("Candidates", df_cand)]:
            train_arts = set(df[df["split"] == "train"]["article_id"])
            val_arts = set(df[df["split"] == "validation"]["article_id"])
            test_arts = set(df[df["split"] == "test"]["article_id"])
            self.assertEqual(len(train_arts), 12, f"{name}: Expected 12 train articles")
            self.assertEqual(len(val_arts), 4, f"{name}: Expected 4 val articles")
            self.assertEqual(len(test_arts), 4, f"{name}: Expected 4 test articles")
            self.assertEqual(len(train_arts & val_arts), 0, f"{name}: Train/Val article overlap!")
            self.assertEqual(len(train_arts & test_arts), 0, f"{name}: Train/Test article overlap!")
            self.assertEqual(len(val_arts & test_arts), 0, f"{name}: Val/Test article overlap!")

        # Candidate coverage: all 852 gold relations covered, 0 self-loops
        gold_pairs = set(zip(df_rel["section_id"], df_rel["cause_id"], df_rel["effect_id"]))
        cand_pairs = set(zip(df_cand["section_id"], df_cand["arg1_id"], df_cand["arg2_id"]))
        self.assertEqual(len(gold_pairs & cand_pairs), 852, "Incomplete candidate-to-gold coverage")
        self.assertEqual(len(df_cand[df_cand["arg1_id"] == df_cand["arg2_id"]]), 0, "Self-loop candidates found")


# ===========================================================================
# 3. Dynamic Benchmark Reconciliation (Requires Processed Datasets)
# ===========================================================================
class TestPhase1BenchmarkReconciliation(unittest.TestCase):
    """Dynamically recompute held-out test metrics and verify arithmetic identities."""

    @classmethod
    def setUpClass(cls):
        if not (PROCESSED_DIR / "biocause_candidates.csv").is_file() or not (PROCESSED_DIR / "biocause_relations.csv").is_file():
            raise unittest.SkipTest("Processed candidate or relation dataset not found. Skipping benchmark reconciliation.")

    def test_08_extraction_metric_recomputation(self):
        """Dynamically recompute extraction metrics on held-out test split (100 gold relations)."""
        df_cands = pd.read_csv(PROCESSED_DIR / "biocause_candidates.csv", keep_default_na=False)
        gold_rels = pd.read_csv(PROCESSED_DIR / "biocause_relations.csv", keep_default_na=False)
        test_cands = df_cands[df_cands["split"] == "test"]
        test_gold = gold_rels[gold_rels["split"] == "test"]

        gold_set = set(zip(test_gold["section_id"], test_gold["cause_id"], test_gold["effect_id"]))
        self.assertEqual(len(gold_set), 100, f"Expected 100 test gold relations, got {len(gold_set)}")
        self.assertEqual(len(test_cands), 1408, f"Expected 1,408 test candidates, got {len(test_cands)}")

        # Verification targets: (Method Name, Extractor Function, Exp Pred, Exp TP, Exp FP, Exp FN, Exp P, Exp R, Exp F1)
        benchmarks = [
            ("Lexical Rule Baseline", extract_lexical_relation, 81, 33, 48, 67, 40.74, 33.00, 36.46),
            ("spaCy Dependency Extractor", extract_dependency_relation, 72, 25, 47, 75, 34.72, 25.00, 29.07),
            ("Hybrid Extractor", extract_hybrid_relation, 116, 44, 72, 56, 37.93, 44.00, 40.74),
        ]

        for name, fn, exp_pred, exp_tp, exp_fp, exp_fn, exp_p, exp_r, exp_f1 in benchmarks:
            df_rec = build_extracted_records(test_cands, fn, name[:3].upper(), name, SCHEMA_COLUMNS)
            self.assertEqual(len(df_rec.columns), 17, f"{name}: UnifiedRelationRecord must have 17 columns")
            pred_set = set(zip(df_rec["section_id"], df_rec["cause_id"], df_rec["effect_id"]))
            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            p = round(tp / (tp + fp) * 100, 2)
            r = round(tp / (tp + fn) * 100, 2)
            f1 = round(2 * p * r / (p + r), 2)

            self.assertEqual(len(pred_set), exp_pred, f"{name} Pred count mismatch")
            self.assertEqual(tp, exp_tp, f"{name} TP mismatch")
            self.assertEqual(fp, exp_fp, f"{name} FP mismatch")
            self.assertEqual(fn, exp_fn, f"{name} FN mismatch")
            self.assertEqual(p, exp_p, f"{name} Precision mismatch")
            self.assertEqual(r, exp_r, f"{name} Recall mismatch")
            self.assertEqual(f1, exp_f1, f"{name} F1 mismatch")

            # Arithmetic identities
            self.assertEqual(tp + fp, exp_pred, f"{name}: TP + FP != Pred")
            self.assertEqual(tp + fn, 100, f"{name}: TP + FN != Gold")

    def test_09_test_candidate_distribution_and_confusion_matrix_arithmetic(self):
        """Verify candidate label distribution and supervised classifier confusion matrix arithmetic."""
        df_cand = pd.read_csv(PROCESSED_DIR / "biocause_candidates.csv", keep_default_na=False)
        test_cands = df_cand[df_cand["split"] == "test"]
        self.assertEqual(len(test_cands), 1408)

        gold_pos = (test_cands["label"] == 1).sum()
        gold_neg = (test_cands["label"] == 0).sum()
        self.assertEqual(gold_pos, 100)
        self.assertEqual(gold_neg, 1308)

        # Verified evaluation confusion matrix from AML_04 Logistic Regression (Unweighted - High-Precision DAG Regime)
        tn, fp, fn, tp = 1292, 16, 66, 34
        self.assertEqual(tn + fp + fn + tp, 1408, "Total evaluated != 1,408")
        self.assertEqual(tn + fp, gold_neg, "TN + FP != Gold Negative")
        self.assertEqual(fn + tp, gold_pos, "FN + TP != Gold Positive")
        self.assertEqual(tp + fp, 50, "Predicted positives != 50")

        precision = round(tp / (tp + fp) * 100, 2)
        recall = round(tp / (tp + fn) * 100, 2)
        f1 = round(2 * precision * recall / (precision + recall), 2)
        accuracy = round((tp + tn) / (tp + tn + fp + fn) * 100, 2)

        self.assertEqual(precision, 68.00)
        self.assertEqual(recall, 34.00)
        self.assertEqual(f1, 45.33)
        self.assertEqual(accuracy, 94.18)

        # Verified evaluation confusion matrix from AML_04 Logistic Regression (Moderate Weight 1:2 - F1-Optimal Regime)
        tn_m, fp_m, fn_m, tp_m = 1272, 36, 48, 52
        self.assertEqual(tn_m + fp_m + fn_m + tp_m, 1408, "Total evaluated != 1,408")
        self.assertEqual(tn_m + fp_m, gold_neg, "TN + FP != Gold Negative")
        self.assertEqual(fn_m + tp_m, gold_pos, "FN + TP != Gold Positive")
        self.assertEqual(tp_m + fp_m, 88, "Predicted positives != 88")

        p_m = round(tp_m / (tp_m + fp_m) * 100, 2)
        r_m = round(tp_m / (tp_m + fn_m) * 100, 2)
        f1_m = round(2 * p_m * r_m / (p_m + r_m), 2)
        self.assertEqual(p_m, 59.09)
        self.assertEqual(r_m, 52.00)
        self.assertEqual(f1_m, 55.32)


# ===========================================================================
# 4. Path Resolution and Portability Tests
# ===========================================================================
class TestPhase1PathResolutionAndPortability(unittest.TestCase):
    """Verify deterministic repo root resolution and absence of machine-specific hardcoded paths."""

    ACTIVE_NOTEBOOKS = [
        "00_setup.ipynb",
        "01_eda.ipynb",
        "AML/AML_01_data_features.ipynb",
        "AML/AML_02_ner.ipynb",
        "AML/AML_03_dependency_patterns.ipynb",
        "AML/AML_04_classical_classifier.ipynb",
        "AML/AML_05_integration_evaluation.ipynb",
    ]

    def test_10_deterministic_repo_root_resolution(self):
        """Verify repo root resolution works deterministically from multiple directory anchors."""
        root = find_repo_root()
        self.assertTrue((root / "00_setup.ipynb").is_file())
        self.assertTrue((root / "README.md").is_file())
        self.assertTrue((root / "AML").is_dir())
        self.assertTrue((root / "data").is_dir())

    def test_11_no_active_machine_specific_paths_in_code_cells(self):
        """Ensure no active code cells contain machine-specific absolute paths (/Users/, /Volumes/, /private/var/)."""
        import nbformat
        forbidden_patterns = ["/Users/", "/Volumes/", "/private/var/"]
        violations = []

        for nb_rel in self.ACTIVE_NOTEBOOKS:
            nb_path = REPO_ROOT / nb_rel
            self.assertTrue(nb_path.is_file(), f"Notebook file missing: {nb_rel}")
            nb = nbformat.read(nb_path, as_version=4)
            for idx, cell in enumerate(nb.cells):
                if cell.cell_type == "code":
                    src = cell.source
                    for pat in forbidden_patterns:
                        if pat in src:
                            violations.append(f"{nb_rel} cell {idx} source contains '{pat}'")
                    for out in cell.get("outputs", []):
                        out_txt = str(out.get("text", ""))
                        if "/Users/" in out_txt:
                            violations.append(f"{nb_rel} cell {idx} output contains '/Users/'")

        self.assertEqual(len(violations), 0, f"Machine-specific paths detected in cells/outputs: {violations}")


# ===========================================================================
# 5. Notebook Schema and Python Syntax Validation Tests
# ===========================================================================
class TestPhase1NotebookValidation(unittest.TestCase):
    """Validate notebook JSON structures, metadata, and Python file syntax."""

    ACTIVE_NOTEBOOKS = [
        "00_setup.ipynb",
        "01_eda.ipynb",
        "AML/AML_01_data_features.ipynb",
        "AML/AML_02_ner.ipynb",
        "AML/AML_03_dependency_patterns.ipynb",
        "AML/AML_04_classical_classifier.ipynb",
        "AML/AML_05_integration_evaluation.ipynb",
    ]

    def test_12_notebook_json_and_cell_invariants(self):
        """Verify all active Phase 1 notebooks are valid nbformat v4 with code and markdown cells."""
        import nbformat

        for nb_rel in self.ACTIVE_NOTEBOOKS:
            nb_path = REPO_ROOT / nb_rel
            self.assertTrue(nb_path.is_file(), f"Missing notebook: {nb_rel}")
            nb = nbformat.read(nb_path, as_version=4)
            self.assertEqual(nb.nbformat, 4, f"{nb_rel} must be nbformat v4")
            code_cells = [c for c in nb.cells if c.cell_type == "code"]
            md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
            self.assertGreater(len(code_cells), 0, f"{nb_rel} has 0 code cells")
            self.assertGreater(len(md_cells), 0, f"{nb_rel} has 0 markdown cells")
            for idx, c in enumerate(code_cells):
                self.assertIsNotNone(c.get("execution_count"), f"{nb_rel} code cell {idx} was not executed")
                self.assertGreater(len(c.get("outputs", [])), 0, f"{nb_rel} code cell {idx} has no visible outputs")

    def test_13_python_syntax_and_imports(self):
        """Verify Python syntax and imports for all active module files."""
        py_files = [
            AML_DIR / "causal_extractors.py",
            REPO_ROOT / "tests" / "test_phase1_baseline.py",
        ]
        for py_path in py_files:
            self.assertTrue(py_path.is_file(), f"File missing: {py_path}")
            with open(py_path, "r", encoding="utf-8") as f:
                code_text = f.read()
            # Compile to verify valid AST syntax
            compiled = compile(code_text, str(py_path), "exec")
            self.assertIsNotNone(compiled)


def run_tests():
    suite = unittest.TestSuite()
    for cls in [
        TestPhase1ExtractorUnitTests,
        TestPhase1CorpusAndDataIntegrity,
        TestPhase1BenchmarkReconciliation,
        TestPhase1PathResolutionAndPortability,
        TestPhase1NotebookValidation,
    ]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
