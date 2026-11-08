"""DL Pipeline Utilities (DL_01 to DL_05).

Shared module providing reusable helper signatures, configuration constants,
and standardization utilities for Phase 2 Deep Learning models.

Standard 17-column UnifiedRelationRecord schema is preserved to maintain full
interoperability with Phase 1 NetworkX DAG acyclicity enforcement and pgmpy Bayesian
networks.

Note:
    This file outlines module structure, type signatures, and documentation.
    Actual model execution logic is implemented in the respective DL notebooks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import sys
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ==============================================================================
# Environment & Repository Root Discovery
# ==============================================================================

def find_repo_root() -> Path:
    """Locate repository root portably across local, Colab, and Kaggle execution.

    Searches current working directory and parent directories for the presence of
    both 'AML/causal_extractors.py' and 'DL/dl_utils.py'.

    Returns:
        Path to the repository root directory.
    """
    curr = Path(os.getcwd()).resolve()
    for candidate in [curr] + list(curr.parents):
        if (candidate / "AML" / "causal_extractors.py").exists() and (
            candidate / "DL" / "dl_utils.py"
        ).exists():
            return candidate
    return curr


def setup_repo_imports() -> Path:
    """Configure sys.path with repository root and subdirectories portably.

    Returns:
        Path to resolved repository root.
    """
    root = find_repo_root()
    for p in [root, root / "AML", root / "DL"]:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    return root


# ==============================================================================
# Standard Schema & Constant Definitions
# ==============================================================================

# Standard 17-column UnifiedRelationRecord schema (inherited from Phase 1 AML)
UNIFIED_RELATION_SCHEMA: List[str] = [
    "relation_id",
    "article_id",
    "section_id",
    "cause_id",
    "cause_text",
    "cause_start",
    "cause_end",
    "effect_id",
    "effect_text",
    "effect_start",
    "effect_end",
    "direction",
    "confidence",
    "extraction_method",
    "evidence_text",
    "provenance",
    "split",
]

# Standard Transformer Checkpoint Identifiers
DISTILBERT_CHECKPOINT: str = "distilbert-base-uncased"
PUBMEDBERT_CHECKPOINT: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# Entity Boundary Marker Tokens for BioCause Relation Extraction
ENTITY_MARKERS: List[str] = ["[E1]", "[/E1]", "[E2]", "[/E2]"]

# Canonical Dataset Split Sizes
CORR2CAUSE_SPLIT_COUNTS: Dict[str, int] = {
    "train": 205734,
    "validation": 1076,
    "test": 1162,
}

BIOCAUSE_ARTICLE_SPLITS: Dict[str, int] = {
    "train": 12,
    "validation": 4,
    "test": 4,
}

# BioCause Candidate Split Counts and Positive Counts (from candidate generation)
BIOCAUSE_CANDIDATE_COUNTS: Dict[str, Dict[str, int]] = {
    "train": {"total": 10086, "positive": 621, "negative": 9465},
    "validation": {"total": 1914, "positive": 131, "negative": 1783},
    "test": {"total": 1408, "positive": 100, "negative": 1308},
}

# Primary Phase 1 AML Baseline Input for Hybrid Integration
PRIMARY_AML_CLASSIFIER_NAME: str = "Logistic Regression (Moderate Weight 1:2)"

# Conservative Initial Hardware & Training Defaults
DEFAULT_BIOCAUSE_BATCH_SIZE: int = 8
DEFAULT_BIOCAUSE_GRAD_ACCUM: int = 2
DEFAULT_DISTILBERT_BATCH_SIZE: int = 16
DEFAULT_DISTILBERT_GRAD_ACCUM: int = 1
DEFAULT_CHECKPOINT_LIMIT: int = 1


# ==============================================================================
# 1. Seed & Determinism
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, PyTorch (CPU & GPU/MPS) for reproducibility.

    Args:
        seed: Integer random seed value. Defaults to 42.

    Implementation Requirements:
        - Set random.seed(seed)
        - Set np.random.seed(seed)
        - Set torch.manual_seed(seed)
        - Set torch.cuda.manual_seed_all(seed) if CUDA is available
        - Configure torch.backends.cudnn.deterministic = True
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.warning("PyTorch not available; skipping torch seed configuration.")


# ==============================================================================
# 2. Hardware / Device Detection
# ==============================================================================

def get_device() -> str:
    """Detect available compute hardware (NVIDIA CUDA, Apple MPS, or CPU).

    Returns:
        String identifier ('cuda', 'mps', or 'cpu').

    Implementation Requirements:
        - Check torch.cuda.is_available() -> 'cuda'
        - Check torch.backends.mps.is_available() -> 'mps'
        - Fallback to 'cpu'
        - Log device name and available memory
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available; falling back to CPU.")
        return "cpu"

    if torch.cuda.is_available():
        device = "cuda"
        try:
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            logger.info(f"CUDA device detected: {gpu_name} ({mem_gb:.1f} GB)")
        except Exception:
            logger.info("CUDA device detected.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple MPS device detected.")
    else:
        device = "cpu"
        logger.info("No GPU detected; using CPU.")

    return device


# ==============================================================================
# 3. Dataset Validation
# ==============================================================================

def validate_dataset_schema(
    df: Any,
    expected_columns: List[str],
    dataset_name: str = "Dataset",
) -> bool:
    """Validate that an input DataFrame contains all required columns without missing schema fields.

    Args:
        df: pandas DataFrame or Dataset to validate.
        expected_columns: List of required column names.
        dataset_name: Human-readable dataset name for error reporting.

    Returns:
        True if all expected columns are present.

    Raises:
        ValueError: If any expected columns are missing from the DataFrame.

    Implementation Requirements:
        - Check set(expected_columns).issubset(df.columns)
        - Report missing columns clearly
        - Check for unexpected null values in critical key fields (e.g., candidate_id, split, label)
    """
    # Support both pandas DataFrame and HuggingFace Dataset objects
    if hasattr(df, "column_names"):
        # HuggingFace Dataset object
        columns = set(df.column_names)
    elif hasattr(df, "columns"):
        # pandas DataFrame
        columns = set(df.columns)
    else:
        raise TypeError(
            f"{dataset_name}: Expected pandas DataFrame or HuggingFace Dataset, "
            f"got {type(df).__name__}."
        )

    expected_set = set(expected_columns)
    missing = expected_set - columns
    if missing:
        raise ValueError(
            f"{dataset_name}: Missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(columns)}."
        )

    # Check for null values in critical key fields (pandas DataFrames only)
    if isinstance(df, pd.DataFrame):
        critical_fields = {"candidate_id", "split", "label", "article_id"}
        present_critical = critical_fields & columns
        for field in present_critical:
            null_count = df[field].isna().sum()
            if null_count > 0:
                logger.warning(
                    f"{dataset_name}: Column '{field}' has {null_count} null values."
                )

    return True


def validate_biocause_candidates(df: pd.DataFrame, dataset_name: str = "BioCause") -> Dict[str, Any]:
    """Extended validation for BioCause candidate DataFrames.

    Checks:
        - Required columns present
        - Binary labels (0 or 1)
        - No duplicate candidate_ids
        - Valid article-level split isolation

    Args:
        df: BioCause candidates DataFrame.
        dataset_name: Name for error reporting.

    Returns:
        Dictionary with validation results and summary statistics.
    """
    issues: List[str] = []

    # Check required columns
    biocause_required = [
        "candidate_id", "article_id", "section_id", "label", "split",
        "arg1_id", "arg2_id", "arg1_text", "arg2_text",
    ]
    available = set(df.columns)
    missing = set(biocause_required) - available
    if missing:
        issues.append(f"Missing required columns: {sorted(missing)}")

    # Binary labels
    unique_labels = set(df["label"].dropna().unique())
    if not unique_labels.issubset({0, 1}):
        issues.append(f"Expected binary labels {{0, 1}}, found: {unique_labels}")

    # Duplicate candidate_ids
    if "candidate_id" in df.columns:
        dup_count = df["candidate_id"].duplicated().sum()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate candidate_ids.")

    # Article-level split isolation
    split_articles: Dict[str, set] = {}
    for split_name in df["split"].unique():
        split_articles[split_name] = set(
            df[df["split"] == split_name]["article_id"].unique()
        )

    split_names = list(split_articles.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            overlap = split_articles[split_names[i]] & split_articles[split_names[j]]
            if overlap:
                issues.append(
                    f"Article overlap between {split_names[i]} and {split_names[j]}: {overlap}"
                )

    # Summary statistics
    summary = {}
    for split_name, articles in split_articles.items():
        split_df = df[df["split"] == split_name]
        summary[split_name] = {
            "articles": len(articles),
            "total": len(split_df),
            "positive": int((split_df["label"] == 1).sum()),
            "negative": int((split_df["label"] == 0).sum()),
        }

    if issues:
        raise ValueError(
            f"{dataset_name} validation failed:\n" + "\n".join(f"  - {i}" for i in issues)
        )

    return summary


# ==============================================================================
# 4. Input Formatting & Argument Boundary Markers
# ==============================================================================

def format_biocause_input_with_markers(
    sentence: str,
    arg1_start: int,
    arg1_end: int,
    arg2_start: int,
    arg2_end: int,
    arg1_markers: Tuple[str, str] = ("[E1]", "[/E1]"),
    arg2_markers: Tuple[str, str] = ("[E2]", "[/E2]"),
) -> str:
    """Insert explicit entity boundary markers around cause (Arg1) and effect (Arg2) spans.

    The formatting preserves original argument order in the source text and includes
    the full sentence/section context, not merely between_text.

    Example:
        Raw: "Growth in mild acid promotes transcription of genes activated by PmrA."
        Marked: "[E1] Growth in mild acid [/E1] promotes transcription of genes activated by [E2] PmrA [/E2]."

    Args:
        sentence: Raw source sentence or context text.
        arg1_start: Character start index of argument 1 in sentence.
        arg1_end: Character end index of argument 1 in sentence.
        arg2_start: Character start index of argument 2 in sentence.
        arg2_end: Character end index of argument 2 in sentence.
        arg1_markers: Tuple of (open_tag, close_tag) for Arg1. Defaults to ('[E1]', '[/E1]').
        arg2_markers: Tuple of (open_tag, close_tag) for Arg2. Defaults to ('[E2]', '[/E2]').

    Returns:
        Marked text string with explicit entity boundary indicators.

    Implementation Requirements:
        - Validate that character offsets reference valid positions in sentence
        - Detect and reject or log overlapping argument spans (arg1_start < arg2_end and arg2_start < arg1_end)
        - Determine original argument order (arg1 before arg2 or arg2 before arg1)
        - Insert tags from right to left (higher character offset first) to preserve index alignment
        - Ensure both entity spans remain intact and uncorrupted after formatting
    """
    sent_len = len(sentence)

    # Validate character offsets
    for name, start, end in [
        ("arg1", arg1_start, arg1_end),
        ("arg2", arg2_start, arg2_end),
    ]:
        if start < 0:
            raise ValueError(f"{name}_start ({start}) must be >= 0.")
        if end < start:
            raise ValueError(f"{name}_end ({end}) must be >= {name}_start ({start}).")
        if end > sent_len:
            raise ValueError(
                f"{name}_end ({end}) exceeds sentence length ({sent_len})."
            )

    # Detect overlapping spans (but allow adjacent: arg1_end == arg2_start)
    if arg1_start < arg2_end and arg2_start < arg1_end:
        raise ValueError(
            f"Overlapping argument spans: "
            f"arg1=[{arg1_start}:{arg1_end}], arg2=[{arg2_start}:{arg2_end}]."
        )

    # Build list of (position, tag_text, priority) insertion points.
    # Priority: higher = inserted first (right-to-left).
    # Within same position, close tags come before open tags (so close is to the left).
    # We insert from right to left to preserve offsets.
    # Each insertion: (char_position, tag_text)
    # We must determine which span comes first in text.

    if arg1_start <= arg2_start:
        # Arg1 appears first (or they are adjacent)
        first_start, first_end = arg1_start, arg1_end
        first_open, first_close = arg1_markers
        second_start, second_end = arg2_start, arg2_end
        second_open, second_close = arg2_markers
    else:
        # Arg2 appears first
        first_start, first_end = arg2_start, arg2_end
        first_open, first_close = arg2_markers
        second_start, second_end = arg1_start, arg1_end
        second_open, second_close = arg1_markers

    # Insert from right to left to preserve character offsets
    # Order of insertions (right to left):
    #   1. second_close at second_end
    #   2. second_open at second_start
    #   3. first_close at first_end
    #   4. first_open at first_start

    result = sentence
    result = result[:second_end] + " " + second_close + result[second_end:]
    result = result[:second_start] + second_open + " " + result[second_start:]

    # Adjust first span positions are unchanged because they are before second span
    result = result[:first_end] + " " + first_close + result[first_end:]
    result = result[:first_start] + first_open + " " + result[first_start:]

    return result


def crop_context_around_entities(
    marked_text: str,
    max_tokens: int = 512,
    tokenizer: Any = None,
    margin_tokens: int = 50,
) -> str:
    """Crop context around entity markers for long BioCause examples.

    Preserves both complete argument spans and all four entity markers.
    Avoids blindly truncating from the right. Instead, centers the crop
    window around the midpoint between the two entity spans.

    Args:
        marked_text: Text with [E1], [/E1], [E2], [/E2] markers already inserted.
        max_tokens: Maximum token count allowed (including special tokens).
        tokenizer: A HuggingFace tokenizer instance. If None, falls back to
            character-based approximation (4 chars ≈ 1 token).
        margin_tokens: Extra token budget on each side of the entity span region.

    Returns:
        Cropped text preserving all entity markers and both entity spans.
    """
    if tokenizer is not None:
        tokens = tokenizer.encode(marked_text, add_special_tokens=False)
        if len(tokens) <= max_tokens - 2:  # -2 for [CLS] and [SEP]
            return marked_text
    else:
        # Approximate: 4 chars per token
        approx_tokens = len(marked_text) // 4
        if approx_tokens <= max_tokens - 2:
            return marked_text

    # Find marker positions in the marked text
    e1_open = marked_text.find("[E1]")
    e1_close = marked_text.find("[/E1]")
    e2_open = marked_text.find("[E2]")
    e2_close = marked_text.find("[/E2]")

    if any(pos == -1 for pos in [e1_open, e1_close, e2_open, e2_close]):
        # Markers not found; fall back to simple right truncation
        if tokenizer is not None:
            tokens = tokenizer.encode(marked_text, add_special_tokens=False)
            return tokenizer.decode(tokens[: max_tokens - 2], skip_special_tokens=False)
        return marked_text[: (max_tokens - 2) * 4]

    # Determine the span that covers all markers
    span_start = min(e1_open, e2_open)
    span_end = max(e1_close + len("[/E1]"), e2_close + len("[/E2]"))

    if tokenizer is not None:
        # Token-precise cropping
        char_margin = margin_tokens * 6  # generous char estimate
        crop_start = max(0, span_start - char_margin)
        crop_end = min(len(marked_text), span_end + char_margin)

        cropped = marked_text[crop_start:crop_end]
        tokens = tokenizer.encode(cropped, add_special_tokens=False)

        if len(tokens) > max_tokens - 2:
            # Still too long; decode only max_tokens - 2 tokens centered on the entity region
            entity_region = marked_text[span_start:span_end]
            entity_tokens = tokenizer.encode(entity_region, add_special_tokens=False)
            remaining = max_tokens - 2 - len(entity_tokens)
            if remaining < 0:
                # Entity region itself exceeds budget; return it as-is
                return entity_region
            left_budget = remaining // 2
            right_budget = remaining - left_budget

            # Get tokens before and after entity region
            left_text = marked_text[:span_start]
            right_text = marked_text[span_end:]
            left_tokens = tokenizer.encode(left_text, add_special_tokens=False)
            right_tokens = tokenizer.encode(right_text, add_special_tokens=False)

            left_keep = left_tokens[-left_budget:] if left_budget > 0 else []
            right_keep = right_tokens[:right_budget] if right_budget > 0 else []

            final_tokens = left_keep + entity_tokens + right_keep
            return tokenizer.decode(final_tokens, skip_special_tokens=False)

        return cropped
    else:
        # Character-based approximation
        char_budget = (max_tokens - 2) * 4
        entity_len = span_end - span_start
        remaining = char_budget - entity_len
        if remaining < 0:
            return marked_text[span_start:span_end]
        left_budget = remaining // 2
        right_budget = remaining - left_budget
        crop_start = max(0, span_start - left_budget)
        crop_end = min(len(marked_text), span_end + right_budget)
        return marked_text[crop_start:crop_end]


def prepare_tokenizer_and_inputs(
    checkpoint_name: str,
    max_length: int = 512,
    additional_special_tokens: Optional[List[str]] = None,
) -> Any:
    """Initialize tokenizer and configure special tokens (such as argument markers).

    Args:
        checkpoint_name: HuggingFace model checkpoint (e.g., PUBMEDBERT_CHECKPOINT).
        max_length: Maximum sequence length for truncation/padding.
        additional_special_tokens: Optional list of special tokens to register.

    Returns:
        Configured AutoTokenizer instance.

    Implementation Requirements:
        - Load AutoTokenizer.from_pretrained(checkpoint_name)
        - Add additional_special_tokens via tokenizer.add_special_tokens()
        - Verify tokenizer vocabulary size after adding special tokens
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
    tokenizer.model_max_length = max_length

    if additional_special_tokens:
        # Only add tokens that are not already in the vocabulary
        existing = set(tokenizer.get_vocab().keys())
        new_tokens = [t for t in additional_special_tokens if t not in existing]
        if new_tokens:
            num_added = tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens}
            )
            logger.info(
                f"Added {num_added} special tokens to {checkpoint_name}. "
                f"New vocab size: {len(tokenizer)}."
            )
        else:
            logger.info(
                f"All special tokens already present in {checkpoint_name} vocabulary."
            )

    return tokenizer


# ==============================================================================
# 5. Training Utilities & Loss Weighting
# ==============================================================================

def calculate_dynamic_class_weight(
    n_negative: int,
    n_positive: int,
) -> float:
    """Calculate positive class weight dynamically from training set counts.

    Formula:
        pos_weight = N_negative / N_positive

    Critical Discipline:
        Calculated strictly from training set labels. Never include validation
        or test split labels in class weight calculation.

    Args:
        n_negative: Number of negative instances in the training split.
        n_positive: Number of positive instances in the training split.

    Returns:
        Floating point weight value for the positive class.
    """
    if n_positive <= 0:
        raise ValueError("n_positive must be greater than 0 to compute class weight.")
    if n_negative < 0:
        raise ValueError("n_negative must be non-negative.")
    return float(n_negative) / float(n_positive)


def get_training_arguments(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = DEFAULT_BIOCAUSE_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_BIOCAUSE_GRAD_ACCUM,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    fp16: Optional[bool] = None,
    eval_strategy: str = "epoch",
    save_total_limit: int = DEFAULT_CHECKPOINT_LIMIT,
    logging_dir: Optional[str] = None,
) -> Any:
    """Construct HuggingFace TrainingArguments configured for free GPU tiers (Kaggle T4 / Colab).

    Uses `eval_strategy` conforming to modern Transformers API specifications (>=4.40.0),
    replacing the legacy `evaluation_strategy` argument.

    Args:
        output_dir: Directory where model checkpoints will be stored temporarily.
        num_epochs: Total training epochs.
        batch_size: Per-device train/eval batch size. Defaults to DEFAULT_BIOCAUSE_BATCH_SIZE (8).
        gradient_accumulation_steps: Number of update steps to accumulate before backward pass.
        learning_rate: Initial AdamW learning rate.
        weight_decay: Weight decay regularizer.
        fp16: Whether to enable 16-bit mixed precision (enabled automatically when CUDA is available).
        eval_strategy: Evaluation interval ('epoch' or 'steps'). Modern parameter for evaluation_strategy.
        save_total_limit: Maximum checkpoints retained to prevent disk quota exhaustion.
        logging_dir: Optional directory for tensorboard/logging output.

    Returns:
        transformers.TrainingArguments object.

    Implementation Requirements:
        - Detect fp16 automatically if None: fp16 = (torch.cuda.is_available() and not torch.backends.mps.is_available())
        - Set metric_for_best_model="eval_f1" and load_best_model_at_end=True
    """
    from transformers import TrainingArguments
    import torch

    # Auto-detect fp16 capability
    if fp16 is None:
        fp16 = torch.cuda.is_available() and (
            not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
        )

    if logging_dir is None:
        logging_dir = os.path.join(output_dir, "logs")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        eval_strategy=eval_strategy,
        save_strategy=eval_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=logging_dir,
        logging_steps=50,
        report_to="none",
        seed=42,
    )


# ==============================================================================
# 6. Evaluation Metrics
# ==============================================================================

def compute_classification_metrics(
    eval_pred: Any,
    average: str = "macro",
) -> Dict[str, float]:
    """Compute standard classification metrics from model predictions (logits) and ground truth.

    Computes:
        - Accuracy
        - Precision (macro or binary)
        - Recall (macro or binary)
        - F1-Score (macro or binary)

    Args:
        eval_pred: HuggingFace EvalPrediction namedtuple containing (predictions, label_ids).
        average: Averaging method ('binary' or 'macro').

    Returns:
        Dictionary mapping metric names to floating point scores.

    Implementation Requirements:
        - Extract predicted class: np.argmax(logits, axis=-1)
        - Call sklearn.metrics.precision_recall_fscore_support
        - Call sklearn.metrics.accuracy_score
        - Return dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=average, zero_division=0.0
    )

    # Also compute macro F1 if binary is requested (for completeness)
    result = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    # Add macro_f1 when using binary averaging
    if average == "binary":
        _, _, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0.0
        )
        result["macro_f1"] = float(macro_f1)

    # Confusion matrix
    try:
        cm = confusion_matrix(labels, preds)
        result["confusion_matrix"] = cm.tolist()
    except Exception:
        pass

    return result


def compute_classification_metrics_binary(eval_pred: Any) -> Dict[str, float]:
    """Convenience wrapper for binary classification metrics."""
    return compute_classification_metrics(eval_pred, average="binary")


def compute_classification_metrics_macro(eval_pred: Any) -> Dict[str, float]:
    """Convenience wrapper for macro-averaged classification metrics."""
    return compute_classification_metrics(eval_pred, average="macro")


# ==============================================================================
# 7. Probability Fusion (Hybrid AML + DL)
# ==============================================================================

def fuse_probabilities(
    prob_aml: Any,
    prob_dl: Any,
    alpha: float = 0.5,
) -> Any:
    """Linearly interpolate predicted causal probabilities from AML and DL models.

    Formula:
        P_hybrid = alpha * P_DL + (1 - alpha) * P_AML

    Validation Discipline:
        The blending parameter alpha and classification threshold are selected
        strictly using held-out validation predictions to optimize validation F1.
        No test-set predictions may be used during hyperparameter tuning.

    Args:
        prob_aml: 1D array of predicted probabilities from Phase 1 classical classifier.
        prob_dl: 1D array of predicted probabilities from Phase 2 PubMedBERT model.
        alpha: Weight assigned to deep learning model, where alpha in [0.0, 1.0].
            alpha = 1.0 corresponds to pure DL; alpha = 0.0 corresponds to pure AML.

    Returns:
        1D array of blended causal probability estimates.

    Raises:
        ValueError: If alpha is not within [0.0, 1.0] or array dimensions mismatch.
    """
    prob_aml = np.asarray(prob_aml, dtype=np.float64)
    prob_dl = np.asarray(prob_dl, dtype=np.float64)

    # Validate alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")

    # Validate dimensions
    if prob_aml.ndim != 1:
        raise ValueError(
            f"prob_aml must be 1D, got shape {prob_aml.shape}."
        )
    if prob_dl.ndim != 1:
        raise ValueError(
            f"prob_dl must be 1D, got shape {prob_dl.shape}."
        )

    # Validate matching lengths
    if len(prob_aml) != len(prob_dl):
        raise ValueError(
            f"prob_aml and prob_dl must have matching lengths, "
            f"got {len(prob_aml)} and {len(prob_dl)}."
        )

    # Validate probability ranges
    if np.any(prob_aml < 0.0) or np.any(prob_aml > 1.0):
        raise ValueError("prob_aml contains values outside [0.0, 1.0].")
    if np.any(prob_dl < 0.0) or np.any(prob_dl > 1.0):
        raise ValueError("prob_dl contains values outside [0.0, 1.0].")

    return alpha * prob_dl + (1.0 - alpha) * prob_aml


# ==============================================================================
# 8. Unified Record Export & Graph Formatting
# ==============================================================================

def export_predictions(
    df: Any,
    output_path: str,
    format_type: str = "csv",
) -> None:
    """Safely export model predictions to disk with parent directory creation.

    Args:
        df: pandas DataFrame containing extracted predictions.
        output_path: Target filesystem path for the exported file.
        format_type: Output format ('csv' or 'parquet').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format_type == "csv":
        df.to_csv(output_path, index=False)
    elif format_type == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format_type: '{format_type}'. Use 'csv' or 'parquet'.")

    logger.info(f"Exported {len(df)} rows to {output_path} ({format_type}).")


def build_unified_relation_record(
    candidate_id: str,
    article_id: str,
    section_id: str,
    cause_id: str,
    cause_text: str,
    cause_start: int,
    cause_end: int,
    effect_id: str,
    effect_text: str,
    effect_start: int,
    effect_end: int,
    confidence: float,
    extraction_method: str,
    evidence_text: str,
    split: str,
    direction: str = "cause->effect",
    provenance: str = "BioCause",
) -> Dict[str, Any]:
    """Construct a dictionary conforming to the 17-column UnifiedRelationRecord schema.

    Preserves exact alignment with Phase 1 NetworkX DAG compiler and pgmpy Bayesian
    network requirements.

    The relation_id is derived deterministically from the candidate_id using a
    stable hash prefix to ensure uniqueness and reproducibility.

    Returns:
        Dictionary with keys matching UNIFIED_RELATION_SCHEMA.
    """
    # Deterministic relation ID derived from candidate_id
    relation_id = f"dl_{candidate_id}"

    return {
        "relation_id": relation_id,
        "article_id": article_id,
        "section_id": section_id,
        "cause_id": cause_id,
        "cause_text": cause_text,
        "cause_start": cause_start,
        "cause_end": cause_end,
        "effect_id": effect_id,
        "effect_text": effect_text,
        "effect_start": effect_start,
        "effect_end": effect_end,
        "direction": direction,
        "confidence": confidence,
        "extraction_method": extraction_method,
        "evidence_text": evidence_text,
        "provenance": provenance,
        "split": split,
    }
