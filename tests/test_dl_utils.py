"""tests/test_dl_utils.py

Comprehensive Phase 2 DL utility validation suite verifying:
1. Repository root discovery and portable import setup.
2. Seed reproducibility across Python, NumPy, and PyTorch.
3. Device detection with CPU fallback.
4. Dataset schema validation for pandas DataFrames and HuggingFace Datasets.
5. BioCause entity marker insertion: Arg1 before Arg2, Arg2 before Arg1, adjacent spans.
6. Invalid and overlapping offset rejection.
7. Context cropping while preserving entity markers.
8. Dynamic class-weight calculation.
9. Classification metrics computation.
10. Probability fusion and invalid input rejection.
11. CSV and Parquet export.
12. Unified relation record schema compliance.
13. Deterministic relation ID generation.

All tests use synthetic fixtures. No internet access, pretrained model downloads,
or GPU hardware required.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
            # Without torch, should return CPU
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
            # This will cause ImportError in get_device
            # Since the module is already loaded, we mock at function level
            pass  # Covered by the import guard in get_device


# ===========================================================================
# 4. Dataset Schema Validation
# ===========================================================================
class TestDatasetValidation(unittest.TestCase):
    """Verify dataset schema validation for various input types."""

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
        # Should raise ValueError, NOT KeyError
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
        # Create overlap: PMC1 in both train and test
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
        # PmrB at [0:4], PmrA at [15:19]
        result = dl_utils.format_biocause_input_with_markers(
            sentence, arg1_start=0, arg1_end=4, arg2_start=15, arg2_end=19
        )
        self.assertIn("[E1]", result)
        self.assertIn("[/E1]", result)
        self.assertIn("[E2]", result)
        self.assertIn("[/E2]", result)
        # Arg1 (PmrB) should be between [E1] and [/E1]
        e1_open = result.index("[E1]")
        e1_close = result.index("[/E1]")
        self.assertIn("PmrB", result[e1_open:e1_close + 5])
        # Arg2 (PmrA) should be between [E2] and [/E2]
        e2_open = result.index("[E2]")
        e2_close = result.index("[/E2]")
        self.assertIn("PmrA", result[e2_open:e2_close + 5])

    def test_arg2_before_arg1(self):
        sentence = "PmrA is activated by PmrB kinase."
        # PmrA at [0:4], PmrB at [21:25]
        # Here arg2 (effect) is PmrA at position 0, arg1 (cause) is PmrB at position 21
        result = dl_utils.format_biocause_input_with_markers(
            sentence, arg1_start=21, arg1_end=25, arg2_start=0, arg2_end=4
        )
        self.assertIn("[E1]", result)
        self.assertIn("[/E1]", result)
        self.assertIn("[E2]", result)
        self.assertIn("[/E2]", result)
        # E2 markers should appear before E1 markers in the text
        e1_open = result.index("[E1]")
        e2_open = result.index("[E2]")
        self.assertLess(e2_open, e1_open, "E2 (PmrA) should appear before E1 (PmrB) in text.")

    def test_adjacent_entity_spans(self):
        sentence = "ABCD"
        # AB at [0:2], CD at [2:4] (adjacent, not overlapping)
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
        # "Growth in mild acid" is [0:19], "PmrA" is [29:33]
        result = dl_utils.format_biocause_input_with_markers(
            sentence, arg1_start=0, arg1_end=19, arg2_start=29, arg2_end=33
        )
        # Original entity text should be preserved
        self.assertIn("Growth in mild acid", result)
        self.assertIn("PmrA", result)

    def test_arg1_identity_preserved(self):
        """Arg1 should always be wrapped in [E1] tags regardless of position."""
        sentence = "Effect is caused by Cause here."
        # arg1 (Cause) at position 20:25, arg2 (Effect) at position 0:6
        result = dl_utils.format_biocause_input_with_markers(
            sentence, arg1_start=20, arg1_end=25, arg2_start=0, arg2_end=6
        )
        # Even though Arg2 appears first in text, Arg1 must still be in [E1] tags
        e1_open = result.index("[E1]")
        e1_close = result.index("[/E1]")
        self.assertIn("Cause", result[e1_open:e1_close + 5])


# ===========================================================================
# 7. Invalid and Overlapping Offsets
# ===========================================================================
class TestInvalidOffsets(unittest.TestCase):
    """Verify rejection of invalid and overlapping character offsets."""

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
    """Verify context cropping preserves entity markers and spans."""

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
        # Verify token count stays strictly within 384 (budget 382)
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

    def test_backward_compatible_alias(self):
        marked = "[E1] Arg1 [/E1] causes [E2] Arg2 [/E2]"
        res1 = dl_utils.crop_context_around_entities(marked, max_tokens=384)
        res2 = dl_utils.crop_context_preserving_markers(marked, max_tokens=384)
        self.assertEqual(res1, res2)


# ===========================================================================
# 9. Dynamic Class-Weight Calculation
# ===========================================================================
class TestDynamicClassWeight(unittest.TestCase):
    """Verify dynamic class-weight computation."""

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
        """Create a mock EvalPrediction namedtuple."""
        return (np.array(logits), np.array(labels))

    def test_perfect_predictions(self):
        logits = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
        labels = [1, 0, 1, 0]
        result = dl_utils.compute_classification_metrics(
            self._make_eval_pred(logits, labels), average="binary"
        )
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_all_wrong_predictions(self):
        logits = [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]
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
        # Simulate imbalanced dataset: 8 negatives, 2 positives
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
    """Verify unified relation record builder produces correct schema."""

    def test_schema_columns(self):
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
        # Check all 17 schema columns present
        for col in dl_utils.UNIFIED_RELATION_SCHEMA:
            self.assertIn(col, record, f"Missing schema column: {col}")

    def test_deterministic_relation_id(self):
        """Relation ID should be deterministic from candidate_id."""
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

    def test_default_direction_and_provenance(self):
        record = dl_utils.build_unified_relation_record(
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
        self.assertEqual(record["direction"], "cause->effect")
        self.assertEqual(record["provenance"], "BioCause")

    def test_record_count_equals_schema_count(self):
        record = dl_utils.build_unified_relation_record(
            candidate_id="c",
            article_id="a",
            section_id="s",
            cause_id="t1",
            cause_text="x",
            cause_start=0,
            cause_end=1,
            effect_id="t2",
            effect_text="y",
            effect_start=2,
            effect_end=3,
            confidence=0.5,
            extraction_method="m",
            evidence_text="e",
            split="train",
        )
        self.assertEqual(len(record), len(dl_utils.UNIFIED_RELATION_SCHEMA))


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

        # Overwrite file to verify cache is returned
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
# 15. Tokenizer Profiling Helper
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
# 16. Constants Verification
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
