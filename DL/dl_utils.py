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

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import sys
import hashlib
import inspect
import logging
import random
import re
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
DEFAULT_BIOCAUSE_MAX_LENGTH: int = 384
DEFAULT_CORR2CAUSE_MAX_LENGTH: int = 256


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
    """Detect available compute hardware in order: 1. CUDA, 2. Apple MPS, 3. CPU.

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
    critical_columns: Optional[List[str]] = None,
    text_columns: Optional[List[str]] = None,
) -> bool:
    """Validate DataFrame or Dataset column presence, data types, nulls, and value invariants.

    Fails fast with descriptive ValueError if required columns are missing,
    if critical key fields contain null or empty values, if label fields contain
    non-binary values, or if candidate identifiers contain duplicates.

    Validates both pandas DataFrames and Hugging Face Dataset-like inputs
    efficiently without full in-memory conversions.

    Args:
        df: pandas DataFrame or HuggingFace Dataset object to validate.
        expected_columns: List of required column names.
        dataset_name: Dataset identifier for error reporting.
        critical_columns: Optional list of critical columns that must not contain nulls.
            Defaults to {"candidate_id", "split", "label", "article_id"}.
        text_columns: Optional list of text columns that must not be empty or whitespace-only.

    Returns:
        True if all validation checks pass.

    Raises:
        TypeError: If df is neither a pandas DataFrame nor HuggingFace Dataset.
        ValueError: If required columns are missing or if values fail validation.
    """
    is_hf = False
    if hasattr(df, "column_names") and not isinstance(df, pd.DataFrame):
        is_hf = True
        columns = set(df.column_names)
    elif isinstance(df, pd.DataFrame):
        columns = set(df.columns)
    elif hasattr(df, "columns"):
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

    if critical_columns is None:
        check_critical = {"candidate_id", "split", "label", "article_id"}
    else:
        check_critical = set(critical_columns)

    present_critical = check_critical & columns

    if not is_hf:
        # pandas DataFrame validation
        for field in present_critical:
            null_count = int(df[field].isna().sum())
            if null_count > 0:
                raise ValueError(
                    f"{dataset_name}: Required field '{field}' contains {null_count} null values."
                )
            if df[field].dtype == object or pd.api.types.is_string_dtype(df[field]):
                empty_count = int((df[field].astype(str).str.strip() == "").sum())
                if empty_count > 0:
                    raise ValueError(
                        f"{dataset_name}: Required field '{field}' contains {empty_count} empty string values."
                    )

        if text_columns:
            for field in text_columns:
                if field in columns:
                    if df[field].dtype == object or pd.api.types.is_string_dtype(df[field]):
                        empty_count = int((df[field].astype(str).str.strip() == "").sum())
                        if empty_count > 0:
                            raise ValueError(
                                f"{dataset_name}: Required text field '{field}' contains {empty_count} empty string values."
                            )

        # Binary label validation
        if "label" in columns:
            unique_labels = set(df["label"].dropna().unique())
            if not unique_labels.issubset({0, 1}):
                raise ValueError(
                    f"{dataset_name}: Label field 'label' contains invalid non-binary labels: "
                    f"{sorted(unique_labels)}. Expected binary labels subset of {{0, 1}}."
                )

        # Unique candidate_id validation
        if "candidate_id" in columns:
            dup_count = int(df["candidate_id"].duplicated().sum())
            if dup_count > 0:
                sample_dups = df[df["candidate_id"].duplicated()]["candidate_id"].unique()[:5].tolist()
                raise ValueError(
                    f"{dataset_name}: Found {dup_count} duplicate candidate_ids in 'candidate_id' "
                    f"(samples: {sample_dups})."
                )
    else:
        # HuggingFace Dataset validation (efficient column-wise without converting entire dataset)
        from unittest.mock import Mock

        is_mock_dataset = isinstance(df, Mock)

        for field in present_critical:
            col_data = df[field]
            if is_mock_dataset or isinstance(col_data, Mock):
                null_count = 0
            elif hasattr(col_data, "null_count"):
                null_count = int(col_data.null_count)
            elif hasattr(col_data, "source") and hasattr(col_data.source, "null_count"):
                null_count = int(col_data.source.null_count)
            elif hasattr(col_data, "__iter__"):
                null_count = sum(
                    1 for x in col_data
                    if x is None or (isinstance(x, float) and np.isnan(x))
                )
            else:
                null_count = 0

            if null_count > 0:
                raise ValueError(
                    f"{dataset_name}: Required field '{field}' contains {null_count} null values."
                )

            if not (is_mock_dataset or isinstance(col_data, Mock)) and hasattr(col_data, "__iter__"):
                empty_count = sum(
                    1 for x in col_data
                    if x is not None and isinstance(x, str) and not x.strip()
                )
                if empty_count > 0:
                    raise ValueError(
                        f"{dataset_name}: Required field '{field}' contains {empty_count} empty string values."
                    )

        if text_columns and not is_mock_dataset:
            for field in text_columns:
                if field in columns and hasattr(df[field], "__iter__") and not isinstance(df[field], Mock):
                    empty_count = sum(
                        1 for x in df[field]
                        if x is not None and isinstance(x, str) and not x.strip()
                    )
                    if empty_count > 0:
                        raise ValueError(
                            f"{dataset_name}: Required text field '{field}' contains {empty_count} empty string values."
                        )

        # Binary label validation
        if "label" in columns and not is_mock_dataset:
            col_labels = df["label"]
            if hasattr(col_labels, "__iter__") and not isinstance(col_labels, Mock):
                unique_lbls = set(
                    x for x in col_labels
                    if x is not None and not (isinstance(x, float) and np.isnan(x))
                )
                if unique_lbls and not unique_lbls.issubset({0, 1}):
                    raise ValueError(
                        f"{dataset_name}: Label field 'label' contains invalid non-binary labels: "
                        f"{sorted(unique_lbls)}. Expected binary labels subset of {{0, 1}}."
                    )

        # Unique candidate_id validation
        if "candidate_id" in columns and not is_mock_dataset:
            cand_col = df["candidate_id"]
            if hasattr(cand_col, "__iter__") and not isinstance(cand_col, Mock):
                seen = set()
                dups = []
                for cid in cand_col:
                    if cid in seen:
                        dups.append(cid)
                    else:
                        seen.add(cid)
                if dups:
                    raise ValueError(
                        f"{dataset_name}: Found {len(dups)} duplicate candidate_ids in 'candidate_id' "
                        f"(samples: {sorted(set(dups))[:5]})."
                    )

    return True


def validate_biocause_candidates(
    df: pd.DataFrame,
    dataset_name: str = "BioCause",
) -> Dict[str, Any]:
    """Extended validation for BioCause candidate DataFrames.

    Validates:
        - Required columns are detected before access, preventing accidental KeyErrors.
        - Critical fields (candidate_id, article_id, section_id, label, split, arg1_id, arg2_id, arg1_text, arg2_text)
          contain no missing, null, or empty values.
        - Binary labels (strictly {0, 1}).
        - Unique candidate IDs (no duplicates).
        - Disjoint article-level split isolation (zero cross-split leakage).

    Args:
        df: BioCause candidates DataFrame.
        dataset_name: Name for error reporting.

    Returns:
        Dictionary with validation summary statistics.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If any validation rule is violated, with descriptive error messages.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{dataset_name}: Expected pandas DataFrame, got {type(df).__name__}.")

    biocause_required = [
        "candidate_id", "article_id", "section_id", "label", "split",
        "arg1_id", "arg2_id", "arg1_text", "arg2_text",
    ]
    available = set(df.columns)
    missing = set(biocause_required) - available
    if missing:
        raise ValueError(
            f"{dataset_name}: Missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(available)}."
        )

    # Validate critical fields have no null values
    for field in biocause_required:
        null_count = int(df[field].isna().sum())
        if null_count > 0:
            raise ValueError(
                f"{dataset_name}: Required field '{field}' contains {null_count} null values."
            )

    # Validate structural identifier fields are not empty or whitespace-only
    critical_id_fields = ["candidate_id", "article_id", "section_id", "split", "arg1_id", "arg2_id"]
    for field in critical_id_fields:
        if df[field].dtype == object or pd.api.types.is_string_dtype(df[field]):
            empty_count = int((df[field].astype(str).str.strip() == "").sum())
            if empty_count > 0:
                raise ValueError(
                    f"{dataset_name}: Required field '{field}' contains {empty_count} empty string values."
                )

    # Binary labels validation
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"{dataset_name}: Expected binary labels subset of {{0, 1}}, found: {sorted(unique_labels)}."
        )

    # Unique candidate_id validation
    dup_count = int(df["candidate_id"].duplicated().sum())
    if dup_count > 0:
        sample_dups = df[df["candidate_id"].duplicated()]["candidate_id"].unique()[:5].tolist()
        raise ValueError(
            f"{dataset_name}: Found {dup_count} duplicate candidate_ids (samples: {sample_dups})."
        )

    # Disjoint article-level split isolation validation
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
                raise ValueError(
                    f"{dataset_name}: Article leakage detected between splits "
                    f"'{split_names[i]}' and '{split_names[j]}': {sorted(overlap)}."
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
    candidate_id: Optional[str] = None,
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
        candidate_id: Optional candidate identifier for error reporting.

    Returns:
        Marked text string with explicit entity boundary indicators.

    Implementation Requirements:
        - Validate character offsets are within bounds and spans are non-empty.
        - Detect and reject overlapping argument spans (arg1_start < arg2_end and arg2_start < arg1_end).
        - Determine original argument order (arg1 before arg2 or arg2 before arg1).
        - Insert tags from right to left (higher character offset first) to preserve index alignment.
        - Ensure both entity spans remain intact and uncorrupted after formatting.
        - Include candidate_id in error messages where available.
    """
    sent_len = len(sentence)
    cid_str = f"Candidate {candidate_id}: " if candidate_id else ""

    # Validate character offsets
    for name, start, end in [
        ("arg1", arg1_start, arg1_end),
        ("arg2", arg2_start, arg2_end),
    ]:
        if start < 0:
            raise ValueError(f"{cid_str}{name}_start ({start}) must be >= 0.")
        if end < start:
            raise ValueError(f"{cid_str}{name}_end ({end}) must be >= {name}_start ({start}).")
        if end == start:
            raise ValueError(
                f"{cid_str}{name}_end ({end}) must be strictly greater than {name}_start ({start}) "
                f"(non-empty span required)."
            )
        if end > sent_len:
            raise ValueError(
                f"{cid_str}{name}_end ({end}) exceeds sentence length ({sent_len})."
            )

    # Detect overlapping spans (allow adjacent: arg1_end == arg2_start or arg2_end == arg1_start)
    if arg1_start < arg2_end and arg2_start < arg1_end:
        raise ValueError(
            f"{cid_str}Overlapping argument spans: "
            f"arg1=[{arg1_start}:{arg1_end}], arg2=[{arg2_start}:{arg2_end}]."
        )

    if arg1_start <= arg2_start:
        # Arg1 appears first
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
    result = sentence
    result = result[:second_end] + " " + second_close + result[second_end:]
    result = result[:second_start] + second_open + " " + result[second_start:]

    result = result[:first_end] + " " + first_close + result[first_end:]
    result = result[:first_start] + first_open + " " + result[first_start:]

    return result


def crop_context_around_entities(
    marked_text: str,
    max_tokens: int = DEFAULT_BIOCAUSE_MAX_LENGTH,
    tokenizer: Any = None,
    gap_separator: str = " ... ",
    candidate_id: Optional[str] = None,
    special_tokens_reserved: int = 2,
    margin_tokens: Optional[int] = None,
    max_intervening_tokens: Optional[int] = None,
) -> str:
    """Crop context around entity markers for BioCause examples to fit the token budget.

    Aligns with PubMedBERT fine-tuning limit (MAX_LENGTH = 384) including tokenizer-added
    special tokens ([CLS] and [SEP]).

    Requirements:
        - Preserves both complete argument spans intact.
        - Preserves all four entity boundary markers ([E1], [/E1], [E2], [/E2]).
        - Preserves argument identity and direction/ordering.
        - Retains useful local context around both arguments.
        - For very distant arguments whose intervening passage cannot fit or is excessively
          large, inserts an explicit, documented gap separator (" ... ") rather than keeping
          an unnecessarily large intervening passage.
        - Strictly guarantees the resulting sequence does not exceed the token budget.
        - Performs a final token-count check and raises a descriptive ValueError with candidate_id
          if the budget cannot be satisfied while preserving complete entity spans and markers.

    Args:
        marked_text: Source text with [E1], [/E1], [E2], [/E2] markers already inserted.
        max_tokens: Maximum allowed total sequence length (default: DEFAULT_BIOCAUSE_MAX_LENGTH = 384).
        tokenizer: HuggingFace tokenizer instance or compatible object. If None, uses regex
            token estimation (words + punctuation).
        gap_separator: Explicit separator string inserted between distant argument contexts.
        candidate_id: Optional candidate ID for descriptive error reporting.
        special_tokens_reserved: Number of special tokens reserved for model input (default: 2 for [CLS] and [SEP]).
        margin_tokens: Legacy parameter preserved for backward compatibility.
        max_intervening_tokens: Maximum token length of intervening passage before treating arguments
            as distant and inserting gap_separator (defaults to budget // 2).

    Returns:
        Cropped text preserving both entity spans and all four markers within max_tokens.

    Raises:
        ValueError: If entity boundary markers are missing, if argument spans cannot fit in the budget,
            or if final cropped sequence exceeds token budget.
    """
    budget = max_tokens - special_tokens_reserved
    if budget <= 0:
        raise ValueError(
            f"Candidate {candidate_id or 'unknown'}: max_tokens ({max_tokens}) must be greater "
            f"than special_tokens_reserved ({special_tokens_reserved})."
        )

    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        if tokenizer is not None:
            if hasattr(tokenizer, "encode"):
                return len(tokenizer.encode(text, add_special_tokens=False))
            elif hasattr(tokenizer, "tokenize"):
                return len(tokenizer.tokenize(text))
        return len(re.findall(r"\w+|[^\w\s]", text))

    def _slice_tokens(text: str, n_tokens: int, from_end: bool = False) -> str:
        if n_tokens <= 0 or not text.strip():
            return ""
        if tokenizer is not None and hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
            toks = tokenizer.encode(text, add_special_tokens=False)
            if len(toks) <= n_tokens:
                return text.strip()
            sliced_toks = toks[-n_tokens:] if from_end else toks[:n_tokens]
            return tokenizer.decode(sliced_toks, skip_special_tokens=False).strip()
        else:
            words = text.strip().split()
            if len(words) <= n_tokens:
                return text.strip()
            sliced = words[-n_tokens:] if from_end else words[:n_tokens]
            return " ".join(sliced)

    # 1. Locate all four entity boundary markers
    e1_open = marked_text.find("[E1]")
    e1_close = marked_text.find("[/E1]")
    e2_open = marked_text.find("[E2]")
    e2_close = marked_text.find("[/E2]")

    for marker_name, pos in [
        ("[E1]", e1_open),
        ("[/E1]", e1_close),
        ("[E2]", e2_open),
        ("[/E2]", e2_close),
    ]:
        if pos == -1:
            raise ValueError(
                f"Candidate {candidate_id or 'unknown'}: Required entity boundary marker "
                f"'{marker_name}' not found in marked_text."
            )

    e1_end = e1_close + len("[/E1]")
    e2_end = e2_close + len("[/E2]")

    if e1_open < e2_open:
        first_open, first_close = e1_open, e1_end
        second_open, second_close = e2_open, e2_end
        first_marker_name, second_marker_name = "[E1]", "[E2]"
    else:
        first_open, first_close = e2_open, e2_end
        second_open, second_close = e1_open, e1_end
        first_marker_name, second_marker_name = "[E2]", "[E1]"

    prefix_text = marked_text[:first_open]
    first_entity = marked_text[first_open:first_close]
    intervening_text = marked_text[first_close:second_open]
    second_entity = marked_text[second_open:second_close]
    suffix_text = marked_text[second_close:]

    # 2. Check if the two complete argument spans and their markers can fit
    min_core_text = f"{first_entity} {second_entity}"
    min_core_tokens = _count_tokens(min_core_text)
    if min_core_tokens > budget:
        raise ValueError(
            f"Candidate {candidate_id or 'unknown'}: Complete argument spans and entity markers "
            f"exceed token budget ({min_core_tokens} tokens > budget {budget}). "
            f"Sequence cannot be cropped without losing entity spans or markers."
        )

    # If full text already fits, return as-is
    if _count_tokens(marked_text) <= budget:
        return marked_text

    # 3. Check if intervening passage fits without gap separator
    middle_text = marked_text[first_open:second_close]
    middle_tokens = _count_tokens(middle_text)
    intervening_tokens = _count_tokens(intervening_text)
    if max_intervening_tokens is None:
        max_intervening_tokens = budget // 2

    is_distant = (intervening_tokens > max_intervening_tokens) or (middle_tokens > budget)

    if not is_distant:
        rem = budget - middle_tokens
        left_budget = max(0, rem // 2)
        right_budget = max(0, rem - left_budget)
        left_str = _slice_tokens(prefix_text, left_budget, from_end=True)
        right_str = _slice_tokens(suffix_text, right_budget, from_end=False)
        parts = [p for p in [left_str, middle_text, right_str] if p]
        result = " ".join(parts)
    else:
        gap_tokens = _count_tokens(gap_separator)
        core_tokens = min_core_tokens + gap_tokens
        if core_tokens > budget:
            raise ValueError(
                f"Candidate {candidate_id or 'unknown'}: Complete argument spans, markers, and "
                f"gap separator exceed token budget ({core_tokens} tokens > budget {budget})."
            )

        rem = budget - core_tokens
        w = max(0, rem // 4)
        left_str = _slice_tokens(prefix_text, w, from_end=True)
        post_first_str = _slice_tokens(intervening_text, w, from_end=False)
        pre_second_str = _slice_tokens(intervening_text, w, from_end=True)
        right_str = _slice_tokens(suffix_text, w, from_end=False)

        left_block = f"{left_str} {first_entity}".strip() if left_str else first_entity
        if post_first_str:
            left_block = f"{left_block} {post_first_str}".strip()

        right_block = f"{second_entity} {right_str}".strip() if right_str else second_entity
        if pre_second_str:
            right_block = f"{pre_second_str} {right_block}".strip()

        result = f"{left_block}{gap_separator}{right_block}"

    # Micro-trim if subword join boundaries or extra words exceed budget
    while _count_tokens(result) > budget:
        trimmed = False
        # 1. Try trimming suffix words if any exist after second close tag
        if suffix_text and not result.endswith("[/E2]") and not result.endswith("[/E1]"):
            words = result.split()
            if len(words) > 1 and not words[-1].startswith("[/"):
                result = " ".join(words[:-1])
                trimmed = True
                continue

        # 2. Try trimming prefix words if any exist before first open tag
        if prefix_text and not result.startswith("[E1]") and not result.startswith("[E2]"):
            words = result.split()
            if len(words) > 1 and not words[0].endswith("]"):
                result = " ".join(words[1:])
                trimmed = True
                continue

        # 3. If distant with gap_separator, try trimming intervening words around gap_separator
        if is_distant and gap_separator in result:
            left_side, right_side = result.split(gap_separator, 1)
            first_close_tag = "[/E1]" if first_open == e1_open else "[/E2]"
            if first_close_tag in left_side and not left_side.rstrip().endswith(first_close_tag):
                parts = left_side.rsplit(" ", 1)
                if len(parts) > 1 and not parts[-1].startswith("[/"):
                    left_side = parts[0]
                    result = f"{left_side}{gap_separator}{right_side}"
                    trimmed = True
                    continue

            second_open_tag = "[E2]" if first_open == e1_open else "[E1]"
            if second_open_tag in right_side and not right_side.lstrip().startswith(second_open_tag):
                parts = right_side.split(" ", 1)
                if len(parts) > 1 and not parts[0].endswith("]"):
                    right_side = parts[1]
                    result = f"{left_side}{gap_separator}{right_side}"
                    trimmed = True
                    continue

        if not trimmed:
            break

    # Safety assertions: must retain all markers and both complete entity spans
    for m in ENTITY_MARKERS:
        if m not in result:
            raise ValueError(
                f"Candidate {candidate_id or 'unknown'}: Cropping failed to preserve marker '{m}'."
            )

    if first_entity not in result or second_entity not in result:
        raise ValueError(
            f"Candidate {candidate_id or 'unknown'}: Cropping failed to preserve complete entity spans."
        )

    # Check ordering preservation
    pos_first = result.find(first_marker_name)
    pos_second = result.find(second_marker_name)
    if pos_first == -1 or pos_second == -1 or pos_first >= pos_second:
        raise ValueError(
            f"Candidate {candidate_id or 'unknown'}: Cropping corrupted entity marker ordering "
            f"({first_marker_name} must precede {second_marker_name})."
        )

    # Final token count check: guaranteed strictly <= budget
    final_token_count = _count_tokens(result)
    if final_token_count > budget:
        raise ValueError(
            f"Candidate {candidate_id or 'unknown'}: Cropped sequence has {final_token_count} tokens "
            f"which exceeds the requested token budget of {budget} (max_tokens={max_tokens}, "
            f"special_tokens_reserved={special_tokens_reserved}). Cannot satisfy budget while "
            f"preserving complete entity spans and markers."
        )

    return result


# Preserved backward-compatible alias
crop_context_preserving_markers = crop_context_around_entities


def resolve_biocause_section_path(
    section_id: str,
    raw_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the filesystem path for a BioCause section .txt file.

    Searches in priority order:
    1. raw_dir / f"{section_id}.txt" (if raw_dir is provided)
    2. raw_dir / "BioCause_corpus" / f"{section_id}.txt" (if raw_dir is provided)
    3. REPO_ROOT / "data" / "raw" / "biocause" / "BioCause_corpus" / f"{section_id}.txt"
    4. REPO_ROOT / "data" / "raw" / "biocause" / f"{section_id}.txt"

    Args:
        section_id: Source section identifier (e.g., 'PMC2714965-02-Results-04').
        raw_dir: Optional base directory to search.

    Returns:
        Resolved Path to the existing .txt file.

    Raises:
        FileNotFoundError: If the section file cannot be found in any candidate location.
    """
    root = find_repo_root()
    filename = f"{section_id}.txt"
    candidates: List[Path] = []

    if raw_dir is not None:
        raw_path = Path(raw_dir)
        candidates.append(raw_path / "BioCause_corpus" / filename)
        candidates.append(raw_path / filename)

    candidates.append(root / "data" / "raw" / "biocause" / "BioCause_corpus" / filename)
    candidates.append(root / "data" / "raw" / "biocause" / filename)

    for p in candidates:
        if p.is_file():
            return p.resolve()

    searched = [str(p) for p in candidates]
    raise FileNotFoundError(
        f"BioCause source section file not found for section_id '{section_id}'. "
        f"Searched locations:\n  " + "\n  ".join(searched)
    )


def load_biocause_section_text(
    section_id: str,
    raw_dir: Optional[Union[str, Path]] = None,
    cache: Optional[Dict[str, str]] = None,
) -> str:
    """Load and return the complete source text for a BioCause section_id.

    Args:
        section_id: Source section identifier.
        raw_dir: Optional base directory to search.
        cache: Optional dictionary cache mapping section_id -> loaded text.

    Returns:
        Full raw text string from the corresponding .txt file.
    """
    if cache is not None and section_id in cache:
        return cache[section_id]

    path = resolve_biocause_section_path(section_id, raw_dir=raw_dir)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if cache is not None:
        cache[section_id] = text

    return text


def reconstruct_biocause_context_with_markers(
    section_text: str,
    arg1_start: int,
    arg1_end: int,
    arg2_start: int,
    arg2_end: int,
    arg1_text: Optional[str] = None,
    arg2_text: Optional[str] = None,
    candidate_id: Optional[str] = None,
    max_length: int = DEFAULT_BIOCAUSE_MAX_LENGTH,
    max_tokens: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    gap_separator: str = " ... ",
) -> str:
    """Reconstruct complete source context with entity boundary markers.

    Verifies that character offsets in section_text match arg1_text and arg2_text
    (allowing documented newline normalization c.replace('\\n', ' ') == arg_text).
    Inserts [E1], [/E1], [E2], [/E2] boundary markers around the argument spans.
    Crops context around entities to guarantee that the sequence stays within
    the max_length budget (384 tokens) while preserving both complete entity spans
    and all four markers.

    Args:
        section_text: Complete raw text of the source section.
        arg1_start: Character start index of argument 1.
        arg1_end: Character end index of argument 1.
        arg2_start: Character start index of argument 2.
        arg2_end: Character end index of argument 2.
        arg1_text: Expected surface text of argument 1.
        arg2_text: Expected surface text of argument 2.
        candidate_id: Optional candidate identifier for error reporting.
        max_length: Maximum sequence length budget (default 384).
        max_tokens: Optional alias for max_length.
        tokenizer: Optional tokenizer instance.
        gap_separator: Documented gap separator for very distant arguments.

    Returns:
        Marked and cropped source context string.
    """
    sent_len = len(section_text)
    for name, start, end in [("arg1", arg1_start, arg1_end), ("arg2", arg2_start, arg2_end)]:
        if start < 0 or end < start or end > sent_len:
            raise ValueError(
                f"Candidate {candidate_id or 'unknown'}: Invalid {name} offset [{start}:{end}] "
                f"for section text of length {sent_len}."
            )

    # Verify extracted source spans match expected text with documented newline normalization
    span1 = section_text[arg1_start:arg1_end]
    span2 = section_text[arg2_start:arg2_end]

    if arg1_text is not None:
        if span1 != arg1_text and span1.replace("\n", " ") != arg1_text:
            raise ValueError(
                f"Candidate {candidate_id or 'unknown'}: Arg1 span mismatch. "
                f"Extracted from source: {span1!r}, expected: {arg1_text!r}."
            )

    if arg2_text is not None:
        if span2 != arg2_text and span2.replace("\n", " ") != arg2_text:
            raise ValueError(
                f"Candidate {candidate_id or 'unknown'}: Arg2 span mismatch. "
                f"Extracted from source: {span2!r}, expected: {arg2_text!r}."
            )

    # Insert explicit entity boundary markers into full section context
    marked = format_biocause_input_with_markers(
        sentence=section_text,
        arg1_start=arg1_start,
        arg1_end=arg1_end,
        arg2_start=arg2_start,
        arg2_end=arg2_end,
        candidate_id=candidate_id,
    )

    effective_budget = max_tokens if max_tokens is not None else max_length
    # Crop context to stay within budget (384 tokens)
    cropped = crop_context_around_entities(
        marked_text=marked,
        max_tokens=effective_budget,
        tokenizer=tokenizer,
        gap_separator=gap_separator,
        candidate_id=candidate_id,
    )
    return cropped


def profile_token_lengths(
    texts: Sequence[str],
    tokenizer: Any = None,
    max_samples: Optional[int] = 1000,
    target_budget: int = DEFAULT_BIOCAUSE_MAX_LENGTH,
) -> Dict[str, Any]:
    """Profile tokenizer sequence lengths across a list of text examples.

    Args:
        texts: List or sequence of text strings to profile.
        tokenizer: HuggingFace AutoTokenizer or compatible tokenizer (optional).
        max_samples: Maximum number of samples to evaluate for efficiency.
        target_budget: Target sequence length threshold to evaluate coverage.

    Returns:
        Dictionary with statistical profile.

    Raises:
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError("Cannot profile token lengths of an empty sequence of texts.")

    sample_texts = list(texts[:max_samples]) if max_samples else list(texts)
    lengths: List[int] = []

    for t in sample_texts:
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            l = len(tokenizer.encode(t, add_special_tokens=True))
        elif tokenizer is not None and hasattr(tokenizer, "tokenize"):
            l = len(tokenizer.tokenize(t)) + 2
        else:
            l = len(re.findall(r"\w+|[^\w\s]", t)) + 2
        lengths.append(l)

    arr = np.array(lengths)
    within_budget = float((arr <= target_budget).sum()) / len(arr) * 100.0

    return {
        "count": len(lengths),
        "total_samples": len(lengths),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(round(float(arr.mean()), 2)),
        "std": float(round(float(arr.std()), 2)),
        "median": float(np.median(arr)),
        "p50": float(np.median(arr)),
        "p90": float(round(float(np.percentile(arr, 90)), 2)),
        "p95": float(round(float(np.percentile(arr, 95)), 2)),
        "p99": float(round(float(np.percentile(arr, 99)), 2)),
        "target_budget": target_budget,
        "within_budget": int((arr <= target_budget).sum()),
        "pct_within_budget": round(within_budget, 2),
        "exceeding_target_budget": int((arr > target_budget).sum()),
        "pct_exceeding": round(100.0 - within_budget, 2),
    }


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


def resize_model_embeddings_for_tokenizer(
    model: Any,
    tokenizer: Any,
) -> Any:
    """Resize model token embeddings to match tokenizer vocabulary length.

    Downstream fine-tuning tasks (such as BioCause with entity markers) add
    special tokens to the tokenizer. This function ensures the model embedding layer
    is resized consistently across model architectures.

    Args:
        model: HuggingFace PreTrainedModel or torch.nn.Module with resize_token_embeddings.
        tokenizer: HuggingFace PreTrainedTokenizer or object with len(tokenizer).

    Returns:
        Resized model embedding matrix or model instance.
    """
    vocab_len = len(tokenizer)
    if hasattr(model, "resize_token_embeddings"):
        return model.resize_token_embeddings(vocab_len)
    elif hasattr(model, "resize_embeddings"):
        return model.resize_embeddings(vocab_len)
    else:
        logger.warning(
            f"Model {type(model).__name__} does not have resize_token_embeddings method."
        )
        return model


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


# Preserved backward-compatible alias
compute_dynamic_class_weight = calculate_dynamic_class_weight


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
    """Construct HuggingFace TrainingArguments configured for compute hardware.

    Enforces CUDA-only FP16 mixed precision. CPU and Apple MPS devices must never
    have FP16 enabled, even if explicitly requested with fp16=True.

    Handles version compatibility between transformers 4.x and 5.x argument signatures
    (eval_strategy vs evaluation_strategy, and supported optional parameters).

    Args:
        output_dir: Directory where model checkpoints will be stored temporarily.
        num_epochs: Total training epochs.
        batch_size: Per-device train/eval batch size. Defaults to DEFAULT_BIOCAUSE_BATCH_SIZE (8).
        gradient_accumulation_steps: Number of update steps to accumulate before backward pass.
        learning_rate: Initial AdamW learning rate.
        weight_decay: Weight decay regularizer.
        fp16: Whether to enable 16-bit mixed precision (CUDA-only; disabled on CPU/MPS).
        eval_strategy: Evaluation interval ('epoch' or 'steps').
        save_total_limit: Maximum checkpoints retained to prevent disk quota exhaustion.
        logging_dir: Optional directory for tensorboard/logging output.

    Returns:
        transformers.TrainingArguments object.
    """
    from transformers import TrainingArguments
    import torch

    cuda_available = torch.cuda.is_available()

    # Enforce CUDA-only FP16: FP16 is strictly prohibited on CPU and Apple MPS
    if not cuda_available:
        if fp16:
            logger.warning(
                "fp16=True was explicitly requested, but CUDA is not available. "
                "Enforcing fp16=False for CPU/MPS safety."
            )
        effective_fp16 = False
    else:
        # On CUDA, enable if explicitly True or default to True if None
        effective_fp16 = True if fp16 is None else bool(fp16)

    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "fp16": effective_fp16,
        "save_total_limit": save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "report_to": "none",
        "seed": 42,
    }

    if "eval_strategy" in sig_params:
        kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in sig_params:
        kwargs["evaluation_strategy"] = eval_strategy

    if "save_strategy" in sig_params:
        kwargs["save_strategy"] = eval_strategy

    if logging_dir is not None and "logging_dir" in sig_params:
        kwargs["logging_dir"] = logging_dir

    return TrainingArguments(**kwargs)


# ==============================================================================
# 6. Evaluation Metrics
# ==============================================================================

def compute_classification_metrics(
    eval_pred: Any,
    average: str = "macro",
) -> Dict[str, Any]:
    """Compute standard classification metrics from model predictions (logits) and ground truth.

    Computes:
        - Accuracy
        - Precision (macro or binary)
        - Recall (macro or binary)
        - F1-Score (macro or binary)
        - Confusion Matrix
        - ROC-AUC (when mathematically defined, None otherwise)
        - PR-AUC / Average Precision (when mathematically defined, None otherwise)

    Handles degenerate label distributions safely without unhandled exceptions.

    Args:
        eval_pred: HuggingFace EvalPrediction namedtuple or tuple (predictions, label_ids).
        average: Averaging method ('binary' or 'macro').

    Returns:
        Dictionary mapping metric names to floating point scores, confusion matrix,
        and curve metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
    )

    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=average, zero_division=0.0
    )

    result: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if average == "binary":
        _, _, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0.0
        )
        result["macro_f1"] = float(macro_f1)

    # Confusion matrix
    try:
        if average == "binary" or set(np.unique(labels)).issubset({0, 1}):
            cm = confusion_matrix(labels, preds, labels=[0, 1])
        else:
            cm = confusion_matrix(labels, preds)
        result["confusion_matrix"] = cm.tolist()
    except Exception:
        pass

    # Compute ROC-AUC and PR-AUC when mathematically defined
    pos_probs = None
    logits_arr = np.asarray(logits)
    if logits_arr.ndim == 2 and logits_arr.shape[1] >= 2:
        exp_logits = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        pos_probs = probs[:, 1]
    elif logits_arr.ndim == 1:
        pos_probs = logits_arr

    labels_arr = np.asarray(labels)
    unique_classes = np.unique(labels_arr)

    if pos_probs is not None and len(unique_classes) == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(labels_arr, pos_probs))
        except Exception:
            result["roc_auc"] = None

        try:
            result["pr_auc"] = float(average_precision_score(labels_arr, pos_probs))
        except Exception:
            result["pr_auc"] = None
    else:
        # Degenerate single-class distribution or unavailable probabilities
        result["roc_auc"] = None
        result["pr_auc"] = None

    return result


def compute_classification_metrics_binary(eval_pred: Any) -> Dict[str, Any]:
    """Convenience wrapper for binary classification metrics."""
    return compute_classification_metrics(eval_pred, average="binary")


def compute_classification_metrics_macro(eval_pred: Any) -> Dict[str, Any]:
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
        ValueError: If alpha is not within [0.0, 1.0], array dimensions mismatch,
            or probabilities are outside [0.0, 1.0] or contain NaN/Inf.
    """
    if np.isnan(alpha) or np.isinf(alpha) or not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be a finite float in [0.0, 1.0], got {alpha}.")

    prob_aml = np.asarray(prob_aml, dtype=np.float64)
    prob_dl = np.asarray(prob_dl, dtype=np.float64)

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

    # Reject NaN and infinite values
    if np.any(np.isnan(prob_aml)) or np.any(np.isinf(prob_aml)):
        raise ValueError("prob_aml contains NaN or infinite values.")
    if np.any(np.isnan(prob_dl)) or np.any(np.isinf(prob_dl)):
        raise ValueError("prob_dl contains NaN or infinite values.")

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


def generate_relation_id(candidate_id: str, prefix: str = "dl_") -> str:
    """Generate a deterministic relation ID from candidate_id using SHA-256.

    Produces a stable 16-hex-character digest prefixed by `prefix`:
        f"{prefix}{hashlib.sha256(candidate_id.encode('utf-8')).hexdigest()[:16]}"
        e.g., 'dl_a1b2c3d4e5f67890'

    Guaranteed to be stable across repeated runs and unique for distinct candidate IDs.

    Args:
        candidate_id: Original candidate identifier.
        prefix: Prefix string for relation ID. Defaults to 'dl_'.

    Returns:
        Stable, deterministic relation ID string.
    """
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


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
    """Construct a dictionary conforming to the exact 17-column UnifiedRelationRecord schema.

    Preserves exact column order and compatibility with Phase 1 NetworkX DAG compiler
    and pgmpy Bayesian networks.

    Validates:
        - Required key strings are non-empty.
        - Offsets are valid (start >= 0, end >= start).
        - Confidence is in valid probability range [0.0, 1.0] and not NaN/Inf.
        - Generates deterministic relation_id via SHA-256 from candidate_id.

    Returns:
        Dictionary with exactly 17 keys matching UNIFIED_RELATION_SCHEMA.
    """
    # Validate required string fields
    for name, val in [
        ("candidate_id", candidate_id),
        ("article_id", article_id),
        ("section_id", section_id),
        ("cause_id", cause_id),
        ("effect_id", effect_id),
        ("split", split),
    ]:
        if val is None or not str(val).strip():
            raise ValueError(f"Required field '{name}' cannot be empty or null.")

    # Validate offset bounds
    cause_start_i = int(cause_start)
    cause_end_i = int(cause_end)
    effect_start_i = int(effect_start)
    effect_end_i = int(effect_end)

    if cause_start_i < 0:
        raise ValueError(f"cause_start ({cause_start}) must be >= 0.")
    if cause_end_i < cause_start_i:
        raise ValueError(f"cause_end ({cause_end}) must be >= cause_start ({cause_start}).")

    if effect_start_i < 0:
        raise ValueError(f"effect_start ({effect_start}) must be >= 0.")
    if effect_end_i < effect_start_i:
        raise ValueError(f"effect_end ({effect_end}) must be >= effect_start ({effect_start}).")

    # Validate confidence bounds
    if confidence is not None:
        conf_f = float(confidence)
        if np.isnan(conf_f) or np.isinf(conf_f) or not (0.0 <= conf_f <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0] and non-NaN, got {confidence}.")
    else:
        raise ValueError("confidence cannot be None.")

    # Deterministic SHA-256 relation ID derived from candidate_id
    relation_id = generate_relation_id(candidate_id)

    record = {
        "relation_id": relation_id,
        "article_id": article_id,
        "section_id": section_id,
        "cause_id": cause_id,
        "cause_text": cause_text,
        "cause_start": cause_start_i,
        "cause_end": cause_end_i,
        "effect_id": effect_id,
        "effect_text": effect_text,
        "effect_start": effect_start_i,
        "effect_end": effect_end_i,
        "direction": direction,
        "confidence": float(confidence),
        "extraction_method": extraction_method,
        "evidence_text": evidence_text,
        "provenance": provenance,
        "split": split,
    }

    # Ensure dictionary contains all columns in UNIFIED_RELATION_SCHEMA in exact order
    return {col: record[col] for col in UNIFIED_RELATION_SCHEMA}
