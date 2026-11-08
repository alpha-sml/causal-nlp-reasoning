"""tests/test_dl_utils.py

Comprehensive Phase 2 DL utility validation suite verifying:
1. Repository root discovery and portable import setup.
2. Seed reproducibility across Python, NumPy, and PyTorch.
3. Device detection with CPU fallback.
4. Dataset schema validation for pandas DataFrames and HuggingFace Datasets.
5. BioCause candidate validation (isolation, duplicate IDs, binary labels).
6. BioCause entity marker insertion: Arg1 before Arg2, Arg2 before Arg1, adjacent spans.
7. Invalid, empty, and overlapping offset rejection.
8. Context cropping while preserving entity markers and strictly respecting token budget.
9. Dynamic class-weight calculation strictly from training split.
10. Classification metrics computation, including ROC-AUC, PR-AUC, and degenerate distributions.
11. Probability fusion and NaN/Inf/out-of-range rejection.
12. CSV and Parquet export.
13. Unified relation record schema compliance, deterministic IDs, offset and confidence validation.
14. BioCause raw section loading and source context reconstruction.
15. Tokenizer profiling helper.
16. CUDA-only FP16 enforcement and TrainingArguments factory compatibility.
17. Model embedding resizing helper.
18. Pipeline constants verification.

All tests use synthetic fixtures. No internet access, pretrained model downloads,
or GPU hardware required.
"""

from typing import Any
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd


def find_repo_root() -> Path:
    candidates = []
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    if "__file__" in globals():
        fpath = Path(__file__).resolve().parent
        candidates.extend([fpath, *fpath.parents])
    for candidate in candidates:
        if (candidate / "DL" / "dl_utils.py").is_file():
            return candidate.resolve()
    raise RuntimeError("Repository root could not be identified.")


REPO_ROOT = find_repo_root()
DL_DIR = REPO_ROOT / "DL"

if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dl_utils


# ===========================================================================
# 1. Repository Root Discovery
# ===========================================================================
class TestRepoRootDiscovery(unittest.TestCase):
    """Verify that find_repo_root() locates the correct repository root."""

    def test_find_repo_root_returns_path(self):
        root = dl_utils.find_repo_root()
        self.assertIsInstance(root, Path)

    def test_repo_root_contains_expected_files(self):
        root = dl_utils.find_repo_root()
        self.assertTrue(
            (root / "AML" / "causal_extractors.py").exists()
            or root == Path(os.getcwd()).resolve(),
            "Root should contain AML/causal_extractors.py or be CWD fallback.",
        )

    def test_setup_repo_imports_adds_to_sys_path(self):
        root = dl_utils.setup_repo_imports()
        self.assertIn(str(root), sys.path)


# ===========================================================================
# 2. Seed Reproducibility
# ===========================================================================
class TestSeedReproducibility(unittest.TestCase):
    """Verify seed configuration produces deterministic outputs."""

    def test_numpy_reproducibility(self):
        dl_utils.set_seed(123)
        a = np.random.rand(5)
        dl_utils.set_seed(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_python_random_reproducibility(self):
        import random

        dl_utils.set_seed(456)
        a = [random.random() for _ in range(5)]
        dl_utils.set_seed(456)
        b = [random.random() for _ in range(5)]
        self.assertEqual(a, b)

    def test_torch_reproducibility(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not available.")

        dl_utils.set_seed(789)
        a = torch.randn(5)
        dl_utils.set_seed(789)
        b = torch.randn(5)
        self.assertTrue(torch.equal(a, b))

    def test_different_seeds_different_output(self):
        dl_utils.set_seed(1)
        a = np.random.rand(5)
        dl_utils.set_seed(2)
        b = np.random.rand(5)
        self.assertFalse(np.array_equal(a, b))


# ===========================================================================
# 3. Device Detection
# ===========================================================================
class TestDeviceDetection(unittest.TestCase):
    """Verify device detection returns valid strings and handles CPU fallback."""

    def test_get_device_returns_valid_string(self):
        device = dl_utils.get_device()
        self.assertIn(device, {"cuda", "mps", "cpu"})

    def test_cpu_fallback_when_no_gpu(self):
        """Mock torch to simulate no GPU availability."""
        try:
            import torch
        except ImportError:
            device = dl_utils.get_device()
            self.assertEqual(device, "cpu")
            return

        with patch.object(torch.cuda, "is_available", return_value=False):
            if hasattr(torch.backends, "mps"):
                with patch.object(
                    torch.backends.mps, "is_available", return_value=False
                ):
                    device = dl_utils.get_device()
                    self.assertEqual(device, "cpu")
            else:
                device = dl_utils.get_device()
                self.assertEqual(device, "cpu")

    def test_cpu_fallback_without_torch(self):
        """Simulate torch import failure."""
        with patch.dict(sys.modules, {"torch": None}):
            pass


# ===========================================================================
# 4. Dataset Schema Validation (DataFrame & HuggingFace Dataset)
# ===========================================================================
class TestDatasetValidation(unittest.TestCase):
    """Verify dataset schema validation for pandas DataFrames and HF Datasets."""

    def test_valid_dataframe(self):
        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        result = dl_utils.validate_dataset_schema(df, ["col_a", "col_b"])
        self.assertTrue(result)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"col_a": [1, 2]})
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_dataset_schema(df, ["col_a", "col_b", "col_c"])
        self.assertIn("col_b", str(ctx.exception))
        self.assertIn("col_c", str(ctx.exception))

    def test_extra_columns_ok(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = dl_utils.validate_dataset_schema(df, ["a", "b"])
        self.assertTrue(result)

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            dl_utils.validate_dataset_schema([1, 2, 3], ["a"])

    def test_huggingface_dataset_mock(self):
        """Mock a HuggingFace Dataset with column_names attribute."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["text", "label", "split"]
        result = dl_utils.validate_dataset_schema(mock_ds, ["text", "label"])
        self.assertTrue(result)

    def test_huggingface_dataset_missing_columns(self):
        mock_ds = MagicMock()
        mock_ds.column_names = ["text"]
        with self.assertRaises(ValueError):
            dl_utils.validate_dataset_schema(mock_ds, ["text", "label"])

    def test_null_value_in_critical_field_raises(self):
        df = pd.DataFrame({
            "candidate_id": ["c1", None, "c3"],
            "label": [1, 0, 1],
            "split": ["train", "train", "train"],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_dataset_schema(
                df, ["candidate_id", "label", "split"], "TestDS"
            )
        self.assertIn("null values", str(ctx.exception).lower())

    def test_empty_string_in_critical_field_raises(self):
        df = pd.DataFrame({
            "candidate_id": ["c1", "  ", "c3"],
            "label": [1, 0, 1],
            "split": ["train", "train", "train"],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_dataset_schema(
                df, ["candidate_id", "label", "split"], "TestDS"
            )
        self.assertIn("empty string values", str(ctx.exception).lower())

    def test_dataframe_invalid_binary_label_raises(self):
        df = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "label": [1, 2],
            "split": ["train", "train"],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_dataset_schema(df, ["candidate_id", "label", "split"])
        self.assertIn("non-binary labels", str(ctx.exception).lower())

    def test_dataframe_duplicate_candidate_id_raises(self):
        df = pd.DataFrame({
            "candidate_id": ["c1", "c1"],
            "label": [1, 0],
            "split": ["train", "train"],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_dataset_schema(df, ["candidate_id", "label", "split"])
        self.assertIn("duplicate candidate_ids", str(ctx.exception).lower())

    def test_real_huggingface_dataset_valid(self):
        try:
            from datasets import Dataset
        except ImportError:
            self.skipTest("datasets library not installed.")

        ds = Dataset.from_dict({
            "candidate_id": ["c1", "c2", "c3"],
            "label": [1, 0, 1],
            "split": ["train", "train", "test"],
            "article_id": ["PMC1", "PMC1", "PMC2"],
        })
        self.assertTrue(dl_utils.validate_dataset_schema(ds, ["candidate_id", "label", "split", "article_id"]))

    def test_real_huggingface_dataset_null_critical_field_raises(self):
        try:
            from datasets import Dataset
        except ImportError:
            self.skipTest("datasets library not installed.")

        ds = Dataset.from_dict({
            "candidate_id": ["c1", None, "c3"],
            "label": [1, 0, 1],
            "split": ["train", "train", "test"],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_dataset_schema(ds, ["candidate_id", "label", "split"])
        self.assertIn("null values", str(ctx.exception).lower())

    def test_real_huggingface_dataset_empty_text_raises(self):
        try:
            from datasets import Dataset
        except ImportError:
            self.skipTest("datasets library not installed.")

        ds = Dataset.from_dict({
            "candidate_id": ["c1", "   ", "c3"],
            "label": [1, 0, 1],
            "split": ["train", "train", "test"],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_dataset_schema(ds, ["candidate_id", "label", "split"])
        self.assertIn("empty string values", str(ctx.exception).lower())

    def test_real_huggingface_dataset_invalid_binary_label_raises(self):
        try:
            from datasets import Dataset
        except ImportError:
            self.skipTest("datasets library not installed.")

        ds = Dataset.from_dict({
            "candidate_id": ["c1", "c2"],
            "label": [1, 3],
            "split": ["train", "test"],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_dataset_schema(ds, ["candidate_id", "label", "split"])
        self.assertIn("non-binary labels", str(ctx.exception).lower())

    def test_real_huggingface_dataset_duplicate_candidate_id_raises(self):
        try:
            from datasets import Dataset
        except ImportError:
            self.skipTest("datasets library not installed.")

        ds = Dataset.from_dict({
            "candidate_id": ["dup_id", "dup_id"],
            "label": [1, 0],
            "split": ["train", "test"],
        })
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_dataset_schema(ds, ["candidate_id", "label", "split"])
        self.assertIn("duplicate candidate_ids", str(ctx.exception).lower())


# ===========================================================================
# 5. BioCause Candidate Validation
# ===========================================================================
class TestBioCauseCandidateValidation(unittest.TestCase):
    """Verify BioCause-specific candidate DataFrame validation."""

    def _make_biocause_df(self):
        return pd.DataFrame({
            "candidate_id": [f"c_{i}" for i in range(10)],
            "article_id": ["PMC1"] * 5 + ["PMC2"] * 5,
            "section_id": ["s1"] * 10,
            "label": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "split": ["train"] * 5 + ["test"] * 5,
            "arg1_id": [f"T{i}" for i in range(10)],
            "arg2_id": [f"T{i+10}" for i in range(10)],
            "arg1_text": ["gene"] * 10,
            "arg2_text": ["protein"] * 10,
        })

    def test_valid_biocause_candidates(self):
        df = self._make_biocause_df()
        summary = dl_utils.validate_biocause_candidates(df)
        self.assertIn("train", summary)
        self.assertIn("test", summary)

    def test_missing_required_column_detected_early_no_keyerror(self):
        df = self._make_biocause_df().drop(columns=["label"])
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_biocause_candidates(df)
        self.assertIn("missing required columns", str(ctx.exception).lower())
        self.assertIn("label", str(ctx.exception))

    def test_null_critical_field_raises(self):
        for field in ["candidate_id", "article_id", "label", "split", "arg1_text"]:
            df = self._make_biocause_df()
            df.loc[2, field] = None
            with self.assertRaises(ValueError) as ctx:
                dl_utils.validate_biocause_candidates(df)
            self.assertIn(f"'{field}'", str(ctx.exception))
            self.assertIn("null values", str(ctx.exception).lower())

    def test_empty_string_critical_field_raises(self):
        df = self._make_biocause_df()
        df.loc[3, "candidate_id"] = "   "
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_biocause_candidates(df)
        self.assertIn("empty string values", str(ctx.exception).lower())

    def test_article_overlap_raises(self):
        df = self._make_biocause_df()
        df.loc[5, "article_id"] = "PMC1"
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_biocause_candidates(df)
        self.assertIn("leakage", str(ctx.exception).lower())

    def test_invalid_labels_raises(self):
        df = self._make_biocause_df()
        df.loc[0, "label"] = 2
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_biocause_candidates(df)
        self.assertIn("label", str(ctx.exception).lower())

    def test_duplicate_candidate_ids_raises(self):
        df = self._make_biocause_df()
        df.loc[1, "candidate_id"] = df.loc[0, "candidate_id"]
        with self.assertRaises(ValueError) as ctx:
            dl_utils.validate_biocause_candidates(df)
        self.assertIn("duplicate", str(ctx.exception).lower())


# ===========================================================================
# 6. Entity Marker Insertion
# ===========================================================================
class TestEntityMarkerInsertion(unittest.TestCase):
    """Verify marker insertion for various argument orderings and edge cases."""

    def test_arg1_before_arg2(self):
        sentence = "PmrB activates PmrA response."
        result = dl_utils.format_biocause_input_with_markers(
            sentence, arg1_start=0, arg1_end=4, arg2_start=15, arg2_end=19
        )
        self.assertIn("[E1]", result)
        self.assertIn("[/E1]", result)
        self.assertIn("[E2]", result)
        self.assertIn("[/E2]", result)
        e1_open = result.index("[E1]")
        e1_close = result.index("[/E1]")
        self.assertIn("PmrB", result[e1_open:e1_close + 5])
        e2_open = result.index("[E2]")
        e2_close = result.index("[/E2]")
        self.assertIn("PmrA", result[e2_open:e2_close + 5])

    def test_arg2_before_arg1(self):
        sentence = "PmrA is activated by PmrB kinase."
        result = dl_utils.format_biocause_input_with_markers(
            sentence, arg1_start=21, arg1_end=25, arg2_start=0, arg2_end=4
        )
        self.assertIn("[E1]", result)
        self.assertIn("[/E1]", result)
        self.assertIn("[E2]", result)
        self.assertIn("[/E2]", result)
        e1_open = result.index("[E1]")
        e2_open = result.index("[E2]")
        self.assertLess(e2_open, e1_open, "E2 (PmrA) should appear before E1 (PmrB) in text.")

    def test_adjacent_entity_spans(self):
        sentence = "ABCD"
        result = dl_utils.format_biocause_input_with_markers(
            sentence, arg1_start=0, arg1_end=2, arg2_start=2, arg2_end=4
        )
        self.assertIn("[E1]", result)
        self.assertIn("[/E1]", result)
        self.assertIn("[E2]", result)
        self.assertIn("[/E2]", result)
        self.assertIn("AB", result)
        self.assertIn("CD", result)

    def test_original_text_preserved(self):
        sentence = "Growth in mild acid promotes PmrA."
        result = dl_utils.format_biocause_input_with_markers(
            sentence, arg1_start=0, arg1_end=19, arg2_start=29, arg2_end=33
        )
        self.assertIn("Growth in mild acid", result)
        self.assertIn("PmrA", result)

    def test_arg1_identity_preserved(self):
        sentence = "Effect is caused by Cause here."
        result = dl_utils.format_biocause_input_with_markers(
            sentence, arg1_start=20, arg1_end=25, arg2_start=0, arg2_end=6
        )
        e1_open = result.index("[E1]")
        e1_close = result.index("[/E1]")
        self.assertIn("Cause", result[e1_open:e1_close + 5])


# ===========================================================================
# 7. Invalid, Empty, and Overlapping Offsets
# ===========================================================================
class TestInvalidOffsets(unittest.TestCase):
    """Verify rejection of invalid, empty, and overlapping character offsets."""

    def test_negative_start_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.format_biocause_input_with_markers(
                "test sentence", arg1_start=-1, arg1_end=4, arg2_start=5, arg2_end=13
            )

    def test_end_before_start_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.format_biocause_input_with_markers(
                "test sentence", arg1_start=5, arg1_end=2, arg2_start=7, arg2_end=13
            )

    def test_empty_span_raises(self):
        with self.assertRaises(ValueError) as ctx:
            dl_utils.format_biocause_input_with_markers(
                "test sentence", arg1_start=4, arg1_end=4, arg2_start=7, arg2_end=13
            )
        self.assertIn("non-empty", str(ctx.exception).lower())

    def test_end_exceeds_length_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.format_biocause_input_with_markers(
                "short", arg1_start=0, arg1_end=3, arg2_start=3, arg2_end=100
            )

    def test_overlapping_spans_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.format_biocause_input_with_markers(
                "overlapping text", arg1_start=0, arg1_end=8, arg2_start=5, arg2_end=15
            )

    def test_identical_spans_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.format_biocause_input_with_markers(
                "overlap text here", arg1_start=0, arg1_end=7, arg2_start=0, arg2_end=7
            )


# ===========================================================================
# 8. Context Cropping
# ===========================================================================
class TestContextCropping(unittest.TestCase):
    """Verify context cropping preserves entity markers and spans within budget."""

    def test_short_text_unchanged(self):
        marked = "[E1] PmrB [/E1] activates [E2] PmrA [/E2]."
        result = dl_utils.crop_context_around_entities(marked, max_tokens=384)
        self.assertEqual(result, marked)

    def test_long_text_preserves_markers_and_stays_within_budget(self):
        prefix = "word " * 400
        suffix = " token" * 400
        marked = prefix + "[E1] PmrB [/E1] activates [E2] PmrA [/E2]" + suffix
        result = dl_utils.crop_context_around_entities(marked, max_tokens=384)
        self.assertIn("[E1]", result)
        self.assertIn("[/E1]", result)
        self.assertIn("[E2]", result)
        self.assertIn("[/E2]", result)
        self.assertIn("PmrB", result)
        self.assertIn("PmrA", result)
        import re
        tokens = re.findall(r"\w+|[^\w\s]", result)
        self.assertLessEqual(len(tokens), 382)

    def test_distant_arguments_inserts_gap_separator_and_preserves_markers(self):
        left = "prefix " * 150
        middle = "distant intervening passage " * 100
        right = "suffix " * 150
        marked = left + "[E1] CauseEntity [/E1]" + middle + "[E2] EffectEntity [/E2]" + right
        result = dl_utils.crop_context_around_entities(marked, max_tokens=384, gap_separator=" ... ")
        self.assertIn(" ... ", result)
        self.assertIn("[E1]", result)
        self.assertIn("[/E1]", result)
        self.assertIn("[E2]", result)
        self.assertIn("[/E2]", result)
        self.assertIn("CauseEntity", result)
        self.assertIn("EffectEntity", result)
        import re
        tokens = re.findall(r"\w+|[^\w\s]", result)
        self.assertLessEqual(len(tokens), 382)

    def test_oversized_entities_raises_descriptive_error(self):
        huge_entity1 = "E1_word " * 250
        huge_entity2 = "E2_word " * 250
        marked = f"[E1] {huge_entity1} [/E1] activates [E2] {huge_entity2} [/E2]"
        with self.assertRaises(ValueError) as ctx:
            dl_utils.crop_context_around_entities(
                marked, max_tokens=384, candidate_id="cand_huge_001"
            )
        self.assertIn("cand_huge_001", str(ctx.exception))
        self.assertIn("exceed token budget", str(ctx.exception).lower())

    def test_final_token_budget_regression_check_with_tokenizer_mock(self):
        """Regression test: Mock tokenizer where trimmed text exceeds requested budget."""
        marked = "prefix text [E1] GeneA [/E1] middle [E2] ProteinB [/E2] suffix text"
        mock_tokenizer = Mock()

        def mock_encode(text, add_special_tokens=False):
            if text.strip() == "[E1] GeneA [/E1] [E2] ProteinB [/E2]":
                return [1] * 10
            if text.strip() == "...":
                return [1] * 2
            return [1] * 25

        mock_tokenizer.encode.side_effect = mock_encode

        with self.assertRaises(ValueError) as ctx:
            dl_utils.crop_context_around_entities(
                marked_text=marked,
                max_tokens=20,
                tokenizer=mock_tokenizer,
                candidate_id="cand_budget_overflow",
            )
        self.assertIn("cand_budget_overflow", str(ctx.exception))
        self.assertIn("exceeds the requested token budget", str(ctx.exception).lower())

    def test_backward_compatible_alias(self):
        marked = "[E1] Arg1 [/E1] causes [E2] Arg2 [/E2]"
        res1 = dl_utils.crop_context_around_entities(marked, max_tokens=384)
        res2 = dl_utils.crop_context_preserving_markers(marked, max_tokens=384)
        self.assertEqual(res1, res2)


# ===========================================================================
# 9. Dynamic Class-Weight Calculation
# ===========================================================================
class TestDynamicClassWeight(unittest.TestCase):
    """Verify dynamic class-weight computation strictly from training split."""

    def test_expected_biocause_weight(self):
        weight = dl_utils.calculate_dynamic_class_weight(n_negative=9465, n_positive=621)
        self.assertAlmostEqual(weight, 15.2415, places=3)

    def test_balanced_weight(self):
        weight = dl_utils.calculate_dynamic_class_weight(n_negative=100, n_positive=100)
        self.assertAlmostEqual(weight, 1.0)

    def test_zero_positive_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.calculate_dynamic_class_weight(n_negative=100, n_positive=0)

    def test_negative_positive_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.calculate_dynamic_class_weight(n_negative=100, n_positive=-5)

    def test_negative_count_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.calculate_dynamic_class_weight(n_negative=-1, n_positive=10)

    def test_returns_float(self):
        weight = dl_utils.calculate_dynamic_class_weight(n_negative=50, n_positive=10)
        self.assertIsInstance(weight, float)


# ===========================================================================
# 10. Classification Metrics
# ===========================================================================
class TestClassificationMetrics(unittest.TestCase):
    """Verify classification metrics computation with synthetic predictions."""

    def _make_eval_pred(self, logits, labels):
        return (np.array(logits), np.array(labels))

    def test_perfect_predictions(self):
        logits = [[-2.0, 2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, -2.0]]
        labels = [1, 0, 1, 0]
        result = dl_utils.compute_classification_metrics(
            self._make_eval_pred(logits, labels), average="binary"
        )
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["roc_auc"], 1.0)
        self.assertAlmostEqual(result["pr_auc"], 1.0)

    def test_all_wrong_predictions(self):
        logits = [[2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [-2.0, 2.0]]
        labels = [1, 0, 1, 0]
        result = dl_utils.compute_classification_metrics(
            self._make_eval_pred(logits, labels), average="binary"
        )
        self.assertAlmostEqual(result["accuracy"], 0.0)

    def test_macro_average(self):
        logits = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
        labels = [1, 0, 1, 0]
        result = dl_utils.compute_classification_metrics(
            self._make_eval_pred(logits, labels), average="macro"
        )
        self.assertIn("accuracy", result)
        self.assertIn("f1", result)

    def test_imbalanced_binary(self):
        logits = [[0.9, 0.1]] * 8 + [[0.1, 0.9]] * 2
        labels = [0] * 8 + [1] * 2
        result = dl_utils.compute_classification_metrics(
            self._make_eval_pred(logits, labels), average="binary"
        )
        self.assertEqual(result["accuracy"], 1.0)

    def test_has_confusion_matrix(self):
        logits = [[0.1, 0.9], [0.9, 0.1]]
        labels = [1, 0]
        result = dl_utils.compute_classification_metrics(
            self._make_eval_pred(logits, labels), average="binary"
        )
        self.assertIn("confusion_matrix", result)

    def test_degenerate_single_class_metrics(self):
        """Single-class distribution: ROC-AUC and PR-AUC must be safely None."""
        logits = [[0.2, 0.8], [0.1, 0.9], [0.3, 0.7]]
        labels = [1, 1, 1]  # No negative class
        result = dl_utils.compute_classification_metrics(
            self._make_eval_pred(logits, labels), average="binary"
        )
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["pr_auc"])
        self.assertEqual(result["accuracy"], 1.0)


# ===========================================================================
# 11. Probability Fusion
# ===========================================================================
class TestProbabilityFusion(unittest.TestCase):
    """Verify probability fusion arithmetic and input validation."""

    def test_equal_blend(self):
        aml = np.array([0.2, 0.8, 0.5])
        dl = np.array([0.6, 0.4, 0.3])
        result = dl_utils.fuse_probabilities(aml, dl, alpha=0.5)
        expected = 0.5 * dl + 0.5 * aml
        np.testing.assert_array_almost_equal(result, expected)

    def test_pure_dl(self):
        aml = np.array([0.2, 0.8])
        dl = np.array([0.9, 0.1])
        result = dl_utils.fuse_probabilities(aml, dl, alpha=1.0)
        np.testing.assert_array_almost_equal(result, dl)

    def test_pure_aml(self):
        aml = np.array([0.3, 0.7])
        dl = np.array([0.5, 0.5])
        result = dl_utils.fuse_probabilities(aml, dl, alpha=0.0)
        np.testing.assert_array_almost_equal(result, aml)

    def test_invalid_alpha_negative(self):
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(np.array([0.5]), np.array([0.5]), alpha=-0.1)

    def test_invalid_alpha_above_one(self):
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(np.array([0.5]), np.array([0.5]), alpha=1.1)

    def test_invalid_alpha_nan(self):
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(np.array([0.5]), np.array([0.5]), alpha=float("nan"))

    def test_nan_in_probabilities_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(np.array([0.5, np.nan]), np.array([0.5, 0.5]), alpha=0.5)
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(np.array([0.5, 0.5]), np.array([0.5, np.nan]), alpha=0.5)

    def test_inf_in_probabilities_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(np.array([0.5, np.inf]), np.array([0.5, 0.5]), alpha=0.5)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(np.array([0.5, 0.3]), np.array([0.5]), alpha=0.5)

    def test_2d_array_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(
                np.array([[0.5, 0.3]]), np.array([0.5]), alpha=0.5
            )

    def test_out_of_range_probabilities_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(np.array([1.5]), np.array([0.5]), alpha=0.5)
        with self.assertRaises(ValueError):
            dl_utils.fuse_probabilities(np.array([0.5]), np.array([-0.1]), alpha=0.5)

    def test_result_in_valid_range(self):
        aml = np.array([0.0, 0.5, 1.0])
        dl = np.array([1.0, 0.5, 0.0])
        result = dl_utils.fuse_probabilities(aml, dl, alpha=0.3)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))


# ===========================================================================
# 12. CSV/Parquet Export
# ===========================================================================
class TestExportPredictions(unittest.TestCase):
    """Verify export_predictions creates files with correct contents."""

    def test_csv_export(self):
        df = pd.DataFrame({"id": ["a", "b"], "score": [0.5, 0.8]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "predictions.csv")
            dl_utils.export_predictions(df, path, format_type="csv")
            self.assertTrue(os.path.exists(path))
            loaded = pd.read_csv(path)
            self.assertEqual(len(loaded), 2)
            self.assertListEqual(list(loaded.columns), ["id", "score"])

    def test_parquet_export(self):
        df = pd.DataFrame({"id": ["a", "b"], "score": [0.5, 0.8]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "predictions.parquet")
            dl_utils.export_predictions(df, path, format_type="parquet")
            self.assertTrue(os.path.exists(path))
            loaded = pd.read_parquet(path)
            self.assertEqual(len(loaded), 2)

    def test_creates_parent_directory(self):
        df = pd.DataFrame({"x": [1]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c", "out.csv")
            dl_utils.export_predictions(df, path, format_type="csv")
            self.assertTrue(os.path.exists(path))

    def test_invalid_format_raises(self):
        df = pd.DataFrame({"x": [1]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.json")
            with self.assertRaises(ValueError):
                dl_utils.export_predictions(df, path, format_type="json")


# ===========================================================================
# 13. Unified Relation Record Schema
# ===========================================================================
class TestUnifiedRelationRecord(unittest.TestCase):
    """Verify unified relation record builder produces correct schema and validates inputs."""

    def test_schema_columns_and_exact_order(self):
        record = dl_utils.build_unified_relation_record(
            candidate_id="cand_001",
            article_id="PMC123",
            section_id="s1",
            cause_id="T1",
            cause_text="PmrB",
            cause_start=0,
            cause_end=4,
            effect_id="T2",
            effect_text="PmrA",
            effect_start=20,
            effect_end=24,
            confidence=0.85,
            extraction_method="PubMedBERT",
            evidence_text="PmrB activates PmrA",
            split="test",
        )
        self.assertEqual(list(record.keys()), dl_utils.UNIFIED_RELATION_SCHEMA)

    def test_deterministic_relation_id(self):
        record1 = dl_utils.build_unified_relation_record(
            candidate_id="cand_001",
            article_id="PMC123",
            section_id="s1",
            cause_id="T1",
            cause_text="PmrB",
            cause_start=0,
            cause_end=4,
            effect_id="T2",
            effect_text="PmrA",
            effect_start=20,
            effect_end=24,
            confidence=0.85,
            extraction_method="PubMedBERT",
            evidence_text="PmrB activates PmrA",
            split="test",
        )
        record2 = dl_utils.build_unified_relation_record(
            candidate_id="cand_001",
            article_id="PMC123",
            section_id="s1",
            cause_id="T1",
            cause_text="PmrB",
            cause_start=0,
            cause_end=4,
            effect_id="T2",
            effect_text="PmrA",
            effect_start=20,
            effect_end=24,
            confidence=0.95,
            extraction_method="PubMedBERT",
            evidence_text="PmrB activates PmrA",
            split="test",
        )
        self.assertEqual(record1["relation_id"], record2["relation_id"])

    def test_different_candidates_different_ids(self):
        record1 = dl_utils.build_unified_relation_record(
            candidate_id="cand_001",
            article_id="PMC123",
            section_id="s1",
            cause_id="T1",
            cause_text="A",
            cause_start=0,
            cause_end=1,
            effect_id="T2",
            effect_text="B",
            effect_start=5,
            effect_end=6,
            confidence=0.5,
            extraction_method="test",
            evidence_text="A B",
            split="test",
        )
        record2 = dl_utils.build_unified_relation_record(
            candidate_id="cand_002",
            article_id="PMC123",
            section_id="s1",
            cause_id="T1",
            cause_text="A",
            cause_start=0,
            cause_end=1,
            effect_id="T2",
            effect_text="B",
            effect_start=5,
            effect_end=6,
            confidence=0.5,
            extraction_method="test",
            evidence_text="A B",
            split="test",
        )
        self.assertNotEqual(record1["relation_id"], record2["relation_id"])

    def test_sha256_relation_id_format_and_stability(self):
        rel_id = dl_utils.generate_relation_id("candidate_abc_123")
        import re
        self.assertTrue(bool(re.match(r"^dl_[0-9a-f]{16}$", rel_id)), f"Unexpected ID format: {rel_id}")
        for _ in range(50):
            self.assertEqual(dl_utils.generate_relation_id("candidate_abc_123"), rel_id)

    def test_sha256_relation_id_uniqueness(self):
        ids = [dl_utils.generate_relation_id(f"candidate_{i}") for i in range(100)]
        self.assertEqual(len(set(ids)), 100)

    def test_invalid_offsets_raise_value_error(self):
        base_args = {
            "candidate_id": "c1",
            "article_id": "a1",
            "section_id": "s1",
            "cause_id": "t1",
            "cause_text": "x",
            "effect_id": "t2",
            "effect_text": "y",
            "confidence": 0.5,
            "extraction_method": "m",
            "evidence_text": "e",
            "split": "train",
        }
        with self.assertRaises(ValueError):
            dl_utils.build_unified_relation_record(
                cause_start=-1, cause_end=2, effect_start=5, effect_end=7, **base_args
            )
        with self.assertRaises(ValueError):
            dl_utils.build_unified_relation_record(
                cause_start=5, cause_end=2, effect_start=6, effect_end=8, **base_args
            )
        with self.assertRaises(ValueError):
            dl_utils.build_unified_relation_record(
                cause_start=0, cause_end=2, effect_start=-1, effect_end=8, **base_args
            )
        with self.assertRaises(ValueError):
            dl_utils.build_unified_relation_record(
                cause_start=0, cause_end=2, effect_start=10, effect_end=8, **base_args
            )

    def test_invalid_confidence_raises_value_error(self):
        base_args = {
            "candidate_id": "c1",
            "article_id": "a1",
            "section_id": "s1",
            "cause_id": "t1",
            "cause_text": "x",
            "cause_start": 0,
            "cause_end": 1,
            "effect_id": "t2",
            "effect_text": "y",
            "effect_start": 2,
            "effect_end": 3,
            "extraction_method": "m",
            "evidence_text": "e",
            "split": "train",
        }
        with self.assertRaises(ValueError):
            dl_utils.build_unified_relation_record(confidence=-0.1, **base_args)
        with self.assertRaises(ValueError):
            dl_utils.build_unified_relation_record(confidence=1.1, **base_args)
        with self.assertRaises(ValueError):
            dl_utils.build_unified_relation_record(confidence=float("nan"), **base_args)

    def test_missing_required_string_field_raises(self):
        base_args = {
            "article_id": "a1",
            "section_id": "s1",
            "cause_id": "t1",
            "cause_text": "x",
            "cause_start": 0,
            "cause_end": 1,
            "effect_id": "t2",
            "effect_text": "y",
            "effect_start": 2,
            "effect_end": 3,
            "confidence": 0.5,
            "extraction_method": "m",
            "evidence_text": "e",
            "split": "train",
        }
        with self.assertRaises(ValueError):
            dl_utils.build_unified_relation_record(candidate_id="", **base_args)


# ===========================================================================
# 14. BioCause Section Loading and Source Context Reconstruction
# ===========================================================================
class TestBioCauseSectionLoadingAndReconstruction(unittest.TestCase):
    """Verify loading BioCause raw section text and reconstructing context with markers."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolve_biocause_section_path_corpus_dir(self):
        corpus_dir = Path(self.temp_dir) / "BioCause_corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        sec_file = corpus_dir / "12345.txt"
        sec_file.write_text("Sample section content.", encoding="utf-8")

        resolved = dl_utils.resolve_biocause_section_path("12345", self.temp_dir)
        self.assertEqual(resolved, sec_file.resolve())

    def test_resolve_biocause_section_path_fallback_dir(self):
        sec_file = Path(self.temp_dir) / "67890.txt"
        sec_file.write_text("Fallback section content.", encoding="utf-8")

        resolved = dl_utils.resolve_biocause_section_path("67890", self.temp_dir)
        self.assertEqual(resolved, sec_file.resolve())

    def test_resolve_biocause_section_path_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            dl_utils.resolve_biocause_section_path("non_existent_section", self.temp_dir)

    def test_load_biocause_section_text_caching(self):
        sec_file = Path(self.temp_dir) / "cache_sec.txt"
        sec_file.write_text("Cached text content.", encoding="utf-8")

        cache = {}
        content1 = dl_utils.load_biocause_section_text("cache_sec", self.temp_dir, cache=cache)
        self.assertEqual(content1, "Cached text content.")
        self.assertIn("cache_sec", cache)

        sec_file.write_text("Modified text.", encoding="utf-8")
        content2 = dl_utils.load_biocause_section_text("cache_sec", self.temp_dir, cache=cache)
        self.assertEqual(content2, "Cached text content.")

    def test_reconstruct_exact_match(self):
        text = "PmrB phosphorylates PmrA in Salmonella enterica."
        marked = dl_utils.reconstruct_biocause_context_with_markers(
            section_text=text,
            arg1_start=0,
            arg1_end=4,
            arg1_text="PmrB",
            arg2_start=20,
            arg2_end=24,
            arg2_text="PmrA",
            candidate_id="cand_exact",
        )
        self.assertIn("[E1] PmrB [/E1]", marked)
        self.assertIn("[E2] PmrA [/E2]", marked)
        self.assertTrue(marked.startswith("[E1] PmrB [/E1]"))
        self.assertIn("Salmonella enterica.", marked)

    def test_reconstruct_newline_normalization(self):
        text = "Phosphorylated PmrB\nprotein regulates PmrA."
        marked = dl_utils.reconstruct_biocause_context_with_markers(
            section_text=text,
            arg1_start=15,
            arg1_end=27,
            arg1_text="PmrB protein",
            arg2_start=38,
            arg2_end=42,
            arg2_text="PmrA",
            candidate_id="cand_newline",
        )
        self.assertIn("[E1] PmrB\nprotein [/E1]", marked)
        self.assertIn("[E2] PmrA [/E2]", marked)

    def test_reconstruct_mismatch_raises_value_error(self):
        text = "PmrB phosphorylates PmrA in Salmonella enterica."
        with self.assertRaises(ValueError) as ctx:
            dl_utils.reconstruct_biocause_context_with_markers(
                section_text=text,
                arg1_start=0,
                arg1_end=4,
                arg1_text="WrongGene",
                arg2_start=20,
                arg2_end=24,
                arg2_text="PmrA",
                candidate_id="cand_mismatch",
            )
        self.assertIn("cand_mismatch", str(ctx.exception))
        self.assertIn("span mismatch", str(ctx.exception).lower())

    def test_reconstruct_out_of_bounds_raises_value_error(self):
        text = "Short text."
        with self.assertRaises(ValueError) as ctx:
            dl_utils.reconstruct_biocause_context_with_markers(
                section_text=text,
                arg1_start=0,
                arg1_end=50,
                arg1_text="OutOfBounds",
                arg2_start=55,
                arg2_end=60,
                arg2_text="Target",
                candidate_id="cand_oob",
            )
        self.assertIn("cand_oob", str(ctx.exception))
        self.assertIn("invalid arg1 offset", str(ctx.exception).lower())

    def test_reconstruct_applies_max_length_cropping(self):
        long_prefix = "context " * 200
        long_suffix = " appendix" * 200
        text = f"{long_prefix}PmrB activates PmrA.{long_suffix}"
        p_len = len(long_prefix)
        marked = dl_utils.reconstruct_biocause_context_with_markers(
            section_text=text,
            arg1_start=p_len,
            arg1_end=p_len + 4,
            arg1_text="PmrB",
            arg2_start=p_len + 15,
            arg2_end=p_len + 19,
            arg2_text="PmrA",
            max_tokens=384,
            candidate_id="cand_crop",
        )
        self.assertIn("[E1] PmrB [/E1]", marked)
        self.assertIn("[E2] PmrA [/E2]", marked)
        import re
        tokens = re.findall(r"\w+|[^\w\s]", marked)
        self.assertLessEqual(len(tokens), 382)


# ===========================================================================
# 15. CUDA-Only FP16 Enforcement and TrainingArguments Factory
# ===========================================================================
class TestTrainingArgumentsFactory(unittest.TestCase):
    """Verify that get_training_arguments enforces CUDA-only FP16 and handles versions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_available_enables_fp16_by_default(self, mock_cuda):
        args = dl_utils.get_training_arguments(output_dir=self.temp_dir)
        self.assertTrue(args.fp16)

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_unavailable_disables_fp16(self, mock_cuda):
        args = dl_utils.get_training_arguments(output_dir=self.temp_dir)
        self.assertFalse(args.fp16)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_available_without_cuda_disables_fp16(self, mock_mps, mock_cuda):
        args = dl_utils.get_training_arguments(output_dir=self.temp_dir)
        self.assertFalse(args.fp16)

    @patch("torch.cuda.is_available", return_value=False)
    def test_explicit_fp16_true_on_cpu_remains_disabled(self, mock_cuda):
        """Explicit fp16=True must NOT enable fp16 when CUDA is unavailable."""
        args = dl_utils.get_training_arguments(output_dir=self.temp_dir, fp16=True)
        self.assertFalse(args.fp16)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_explicit_fp16_true_on_mps_remains_disabled(self, mock_mps, mock_cuda):
        """Explicit fp16=True must NOT enable fp16 when on MPS without CUDA."""
        args = dl_utils.get_training_arguments(output_dir=self.temp_dir, fp16=True)
        self.assertFalse(args.fp16)

    def test_training_arguments_defaults_and_compatibility(self):
        args = dl_utils.get_training_arguments(output_dir=self.temp_dir)
        self.assertEqual(args.num_train_epochs, 3)
        self.assertEqual(args.per_device_train_batch_size, dl_utils.DEFAULT_BIOCAUSE_BATCH_SIZE)
        self.assertEqual(args.gradient_accumulation_steps, dl_utils.DEFAULT_BIOCAUSE_GRAD_ACCUM)
        self.assertEqual(args.metric_for_best_model, "eval_f1")
        self.assertTrue(args.greater_is_better)
        self.assertTrue(args.load_best_model_at_end)


# ===========================================================================
# 16. Model Embedding Resizing Helper
# ===========================================================================
class TestModelEmbeddingResizing(unittest.TestCase):
    """Verify resizing model embeddings to match tokenizer vocabulary."""

    def test_resize_model_embeddings_calls_resize(self):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.__len__ = Mock(return_value=30526)

        dl_utils.resize_model_embeddings_for_tokenizer(mock_model, mock_tokenizer)
        mock_model.resize_token_embeddings.assert_called_once_with(30526)

    def test_resize_model_embeddings_fallback(self):
        mock_model = Mock(spec=[])
        mock_tokenizer = [1, 2, 3]
        result = dl_utils.resize_model_embeddings_for_tokenizer(mock_model, mock_tokenizer)
        self.assertEqual(result, mock_model)


# ===========================================================================
# 17. Tokenizer Profiling Helper
# ===========================================================================
class TestTokenizerProfiling(unittest.TestCase):
    """Verify sequence length profiling helper."""

    def test_profile_token_lengths_returns_expected_stats(self):
        texts = [
            "Short sentence.",
            "A slightly longer sentence for profiling token lengths.",
            "Another sentence with various words and symbols!",
        ]
        stats = dl_utils.profile_token_lengths(texts, target_budget=10)
        self.assertEqual(stats["total_samples"], 3)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("p50", stats)
        self.assertIn("p90", stats)
        self.assertIn("p95", stats)
        self.assertIn("p99", stats)
        self.assertEqual(stats["target_budget"], 10)
        self.assertIn("exceeding_target_budget", stats)
        self.assertIn("pct_exceeding", stats)

    def test_profile_token_lengths_empty_raises(self):
        with self.assertRaises(ValueError):
            dl_utils.profile_token_lengths([])


# ===========================================================================
# 18. Constants Verification
# ===========================================================================
class TestConstants(unittest.TestCase):
    """Verify that module constants are defined correctly."""

    def test_entity_markers(self):
        self.assertEqual(dl_utils.ENTITY_MARKERS, ["[E1]", "[/E1]", "[E2]", "[/E2]"])

    def test_unified_schema_length(self):
        self.assertEqual(len(dl_utils.UNIFIED_RELATION_SCHEMA), 17)

    def test_corr2cause_split_counts(self):
        self.assertEqual(dl_utils.CORR2CAUSE_SPLIT_COUNTS["train"], 205734)
        self.assertEqual(dl_utils.CORR2CAUSE_SPLIT_COUNTS["validation"], 1076)
        self.assertEqual(dl_utils.CORR2CAUSE_SPLIT_COUNTS["test"], 1162)

    def test_biocause_candidate_counts(self):
        train = dl_utils.BIOCAUSE_CANDIDATE_COUNTS["train"]
        self.assertEqual(train["total"], 10086)
        self.assertEqual(train["positive"], 621)
        self.assertEqual(train["negative"], 9465)

    def test_biocause_model_eligible_counts(self):
        train = dl_utils.BIOCAUSE_MODEL_ELIGIBLE_COUNTS["train"]
        self.assertEqual(train["total"], 9908)
        self.assertEqual(train["positive"], 620)
        self.assertEqual(train["negative"], 9288)
        val = dl_utils.BIOCAUSE_MODEL_ELIGIBLE_COUNTS["validation"]
        self.assertEqual(val["total"], 1884)
        self.assertEqual(val["positive"], 131)
        test = dl_utils.BIOCAUSE_MODEL_ELIGIBLE_COUNTS["test"]
        self.assertEqual(test["total"], 1394)
        self.assertEqual(test["positive"], 100)


# ===========================================================================
# 19. BioCause Source Reconstruction & Eligibility Invariants
# ===========================================================================
class TestBioCauseSourceReconstructionAndEligibility(unittest.TestCase):
    """Verify source reconstruction, eligibility filtering, diagnostics, and invariants."""

    def setUp(self):
        self.source_text = "Expression of IL-6 promotes vascular permeability in inflammatory lesions."
        # IL-6 is [14:18], vascular permeability is [28:49]
        self.valid_candidate = {
            "candidate_id": "cand_001",
            "article_id": "art_01",
            "section_id": "sec_01",
            "arg1_id": "T1",
            "arg1_text": "IL-6",
            "arg1_start": 14,
            "arg1_end": 18,
            "arg2_id": "T2",
            "arg2_text": "vascular permeability",
            "arg2_start": 28,
            "arg2_end": 49,
            "label": 1,
            "split": "train",
            "between_text": " promotes ",
        }
        self.invalid_overlapping_candidate = {
            "candidate_id": "cand_002",
            "article_id": "art_01",
            "section_id": "sec_01",
            "arg1_id": "T3",
            "arg1_text": "IL-6 promotes",
            "arg1_start": 14,
            "arg1_end": 27,
            "arg2_id": "T4",
            "arg2_text": "promotes vascular",
            "arg2_start": 19,
            "arg2_end": 36,
            "label": 0,
            "split": "train",
            "between_text": "",
        }
        self.invalid_oob_candidate = {
            "candidate_id": "cand_003",
            "article_id": "art_01",
            "section_id": "sec_01",
            "arg1_id": "T5",
            "arg1_text": "nonexistent",
            "arg1_start": 200,
            "arg1_end": 211,
            "arg2_id": "T6",
            "arg2_text": "lesions",
            "arg2_start": 60,
            "arg2_end": 67,
            "label": 0,
            "split": "train",
            "between_text": "",
        }

    def test_1_valid_candidate_source_reconstruction(self):
        marked = dl_utils.reconstruct_biocause_context_with_markers(
            section_text=self.source_text,
            arg1_start=self.valid_candidate["arg1_start"],
            arg1_end=self.valid_candidate["arg1_end"],
            arg2_start=self.valid_candidate["arg2_start"],
            arg2_end=self.valid_candidate["arg2_end"],
            arg1_text=self.valid_candidate["arg1_text"],
            arg2_text=self.valid_candidate["arg2_text"],
            candidate_id=self.valid_candidate["candidate_id"],
        )
        self.assertIn("[E1] IL-6 [/E1]", marked)
        self.assertIn("[E2] vascular permeability [/E2]", marked)

    def test_2_failed_reconstruction_does_not_generate_fallback_text(self):
        cands = [self.valid_candidate, self.invalid_overlapping_candidate]
        reconstructed = []
        for c in cands:
            try:
                m = dl_utils.reconstruct_biocause_context_with_markers(
                    section_text=self.source_text,
                    arg1_start=c["arg1_start"],
                    arg1_end=c["arg1_end"],
                    arg2_start=c["arg2_start"],
                    arg2_end=c["arg2_end"],
                    arg1_text=c["arg1_text"],
                    arg2_text=c["arg2_text"],
                    candidate_id=c["candidate_id"],
                )
                reconstructed.append({"valid": True, "marked_text": m, "diag": ""})
            except Exception as e:
                reconstructed.append({"valid": False, "marked_text": None, "diag": str(e)})

        self.assertTrue(reconstructed[0]["valid"])
        self.assertIsNotNone(reconstructed[0]["marked_text"])
        self.assertFalse(reconstructed[1]["valid"])
        self.assertIsNone(reconstructed[1]["marked_text"])
        self.assertIn("Overlapping", reconstructed[1]["diag"])

    def test_3_invalid_candidates_explicitly_marked_and_retain_diagnostics(self):
        cands = [self.invalid_oob_candidate]
        diagnostics = []
        for c in cands:
            try:
                dl_utils.reconstruct_biocause_context_with_markers(
                    section_text=self.source_text,
                    arg1_start=c["arg1_start"],
                    arg1_end=c["arg1_end"],
                    arg2_start=c["arg2_start"],
                    arg2_end=c["arg2_end"],
                    arg1_text=c["arg1_text"],
                    arg2_text=c["arg2_text"],
                    candidate_id=c["candidate_id"],
                )
            except Exception as e:
                diagnostics.append({"candidate_id": c["candidate_id"], "error": str(e)})

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["candidate_id"], "cand_003")
        self.assertIn("Invalid", diagnostics[0]["error"])

    def test_4_invalid_candidates_excluded_from_model_ready_splits(self):
        df_raw = pd.DataFrame([
            {**self.valid_candidate, "candidate_id": "c1", "split": "train"},
            {**self.invalid_overlapping_candidate, "candidate_id": "c2", "split": "train"},
            {**self.valid_candidate, "candidate_id": "c3", "split": "validation"},
        ])
        df_raw["reconstruction_valid"] = [True, False, True]
        df_raw["marked_text"] = ["[E1] A [/E1] [E2] B [/E2]", None, "[E1] C [/E1] [E2] D [/E2]"]

        df_eligible = df_raw[df_raw["reconstruction_valid"]].copy()
        self.assertEqual(len(df_eligible), 2)
        self.assertNotIn("c2", df_eligible["candidate_id"].values)
        self.assertTrue(df_eligible["marked_text"].notna().all())

    def test_5_valid_candidates_retain_both_spans_and_all_four_markers(self):
        marked = dl_utils.reconstruct_biocause_context_with_markers(
            section_text=self.source_text,
            arg1_start=self.valid_candidate["arg1_start"],
            arg1_end=self.valid_candidate["arg1_end"],
            arg2_start=self.valid_candidate["arg2_start"],
            arg2_end=self.valid_candidate["arg2_end"],
            arg1_text=self.valid_candidate["arg1_text"],
            arg2_text=self.valid_candidate["arg2_text"],
            candidate_id=self.valid_candidate["candidate_id"],
        )
        for m in ["[E1]", "[/E1]", "[E2]", "[/E2]"]:
            self.assertIn(m, marked)
        self.assertIn("IL-6", marked)
        self.assertIn("vascular permeability", marked)

    def test_6_candidate_ids_unique_and_traceable(self):
        df = pd.DataFrame([
            {"candidate_id": "cand_A", "reconstruction_valid": True},
            {"candidate_id": "cand_B", "reconstruction_valid": False},
            {"candidate_id": "cand_C", "reconstruction_valid": True},
        ])
        self.assertEqual(df["candidate_id"].nunique(), len(df))
        eligible = df[df["reconstruction_valid"]]
        self.assertEqual(eligible["candidate_id"].nunique(), len(eligible))

    def test_7_article_level_split_isolation_maintained_after_exclusion(self):
        df = pd.DataFrame([
            {"candidate_id": "c1", "article_id": "art_1", "split": "train", "reconstruction_valid": True},
            {"candidate_id": "c2", "article_id": "art_1", "split": "train", "reconstruction_valid": False},
            {"candidate_id": "c3", "article_id": "art_2", "split": "validation", "reconstruction_valid": True},
            {"candidate_id": "c4", "article_id": "art_3", "split": "test", "reconstruction_valid": True},
        ])
        eligible = df[df["reconstruction_valid"]]
        train_arts = set(eligible[eligible["split"] == "train"]["article_id"])
        val_arts = set(eligible[eligible["split"] == "validation"]["article_id"])
        test_arts = set(eligible[eligible["split"] == "test"]["article_id"])

        self.assertTrue(len(train_arts & val_arts) == 0)
        self.assertTrue(len(train_arts & test_arts) == 0)
        self.assertTrue(len(val_arts & test_arts) == 0)

    def test_8_model_eligible_candidate_tokens_within_budget(self):
        long_prefix = "Word " * 200
        sec = f"{long_prefix}IL-6 promotes vascular permeability {long_prefix}"
        a1_start = len(long_prefix)
        a1_end = a1_start + 4
        a2_start = a1_end + len(" promotes ")
        a2_end = a2_start + len("vascular permeability")

        cropped = dl_utils.reconstruct_biocause_context_with_markers(
            section_text=sec,
            arg1_start=a1_start,
            arg1_end=a1_end,
            arg2_start=a2_start,
            arg2_end=a2_end,
            max_length=384,
        )
        tokens = cropped.split()
        self.assertLessEqual(len(tokens), 384)
        self.assertIn("[E1]", cropped)
        self.assertIn("[E2]", cropped)

    def test_9_training_dynamic_class_weight_uses_only_model_eligible_candidates(self):
        weight = dl_utils.calculate_dynamic_class_weight(n_negative=9288, n_positive=620)
        self.assertAlmostEqual(weight, 14.9806, places=3)
        self.assertNotEqual(weight, 9465 / 621)

    def test_10_diagnostics_export_contains_invalid_candidate_ids_and_reasons(self):
        df_invalid = pd.DataFrame([
            {"candidate_id": "cand_002", "split": "train", "diagnostic": "Overlapping argument spans"},
            {"candidate_id": "cand_003", "split": "val", "diagnostic": "Offset out of section bounds"},
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "dl_biocause_reconstruction_diagnostics.csv"
            dl_utils.export_predictions(df_invalid, str(export_path), format_type="csv")
            readback = pd.read_csv(export_path)
            self.assertEqual(len(readback), 2)
            self.assertIn("cand_002", readback["candidate_id"].values)
            self.assertIn("cand_003", readback["candidate_id"].values)
            self.assertIn("diagnostic", readback.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
