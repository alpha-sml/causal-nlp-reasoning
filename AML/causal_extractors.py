"""causal_extractors.py

Reusable rule-based, syntactic dependency-tree, and hybrid causal relation extractors
for the BioCause dataset and causal NLP benchmark.

This module provides the shared extraction logic used consistently by both:
- AML/AML_03_dependency_patterns.ipynb
- AML/AML_05_integration_evaluation.ipynb

Key capabilities:
1. Lexical pattern matching with active, passive, and discourse cues.
2. spaCy dependency parsing with grammatical role analysis (agent, nsubjpass, etc.).
3. Directional normalization (cause -> effect) across forward and inverted constructions.
4. Linguistic controls: negation pruning, correlation-not-causation filtering,
   epistemic uncertainty detection, self-loop prevention, and empty text handling.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import re
import pandas as pd
import spacy

# ---------------------------------------------------------------------------
# Pattern Definitions
# ---------------------------------------------------------------------------

ACTIVE_CAUSAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("P_ACTIVE_CAUSE", re.compile(r"(?i)\b(?:causes?|caused|causing|is a cause of)\b")),
    ("P_ACTIVE_PRODUCE", re.compile(r"(?i)\b(?:promotes?|promoted|promoting|induces?|induced|inducing|activates?|activated|activating)\b")),
    ("P_ACTIVE_LEAD", re.compile(r"(?i)\b(?:leads? to|led to|leading to|results? in|resulted in|resulting in)\b")),
    ("P_ACTIVE_REGULATE", re.compile(r"(?i)\b(?:triggers?|triggered|triggering|stimulates?|stimulated|stimulating)\b")),
]

PASSIVE_CAUSAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("P_PASSIVE_CAUSE", re.compile(r"(?i)\b(?:is caused by|was caused by|are caused by)\b")),
    ("P_PASSIVE_RESULT", re.compile(r"(?i)\b(?:results from|resulted from|resulting from)\b")),
    ("P_PASSIVE_DUE_TO", re.compile(r"(?i)\b(?:is due to|was due to|are due to|due to)\b")),
    ("P_PASSIVE_MEDIATE", re.compile(r"(?i)\b(?:is mediated by|was mediated by|are mediated by)\b")),
]

DISCOURSE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("P_DISCOURSE_SUGGEST", re.compile(r"(?i)\b(?:suggesting that|indicating that|showing that|which suggests that|suggests that|indicates that)\b")),
    ("P_DISCOURSE_BECAUSE", re.compile(r"(?i)\b(?:because|since|therefore|thus|hence)\b")),
    ("P_DISCOURSE_REQUIRED", re.compile(r"(?i)\b(?:required for|necessary for|essential for)\b")),
]

NEGATION_PATTERN: re.Pattern = re.compile(r"(?i)\b(?:not|never|no|failed to|fails to|unable to|without|neither|nor)\b")
CORRELATION_PATTERN: re.Pattern = re.compile(r"(?i)\b(?:correlated with|associated with|co-occurs with|accompanied by)\b")
UNCERTAINTY_PATTERN: re.Pattern = re.compile(r"(?i)\b(?:may|might|could|suggests|suggesting|possible|hypothesize|hypothesized)\b")

CAUSAL_LEMMAS: Set[str] = {"cause", "induce", "promote", "activate", "lead", "result", "trigger", "stimulate", "mediate"}
NEG_CUES: Set[str] = {"not", "never", "no", "without", "failed", "neither", "nor", "fails", "unable"}

# ---------------------------------------------------------------------------
# Lazy SpaCy Loader
# ---------------------------------------------------------------------------

_NLP_DEP = None

def get_spacy_nlp(model_name: str = "en_core_web_sm"):
    """Lazily load and cache spaCy model with NER disabled for fast parsing."""
    global _NLP_DEP
    if _NLP_DEP is None:
        _NLP_DEP = spacy.load(model_name, disable=["ner"])
    return _NLP_DEP


# ---------------------------------------------------------------------------
# Extractor Implementations
# ---------------------------------------------------------------------------

def extract_lexical_relation(
    arg1_text: str,
    arg2_text: str,
    between_text: str,
    arg1_before_arg2: int = 1,
    max_between_chars: int = 400
) -> Dict[str, Any]:
    """Extract causal relation using lexical patterns and directional argument ordering.
    
    Returns a dictionary containing:
      - is_causal: 1 if any causal cue matches, 0 otherwise
      - pred_directed: 1 if predicted directed causality is from arg1 to arg2, 0 otherwise
      - cause, effect: string names of identified cause and effect
      - direction: 'arg1->arg2' or 'arg2->arg1'
      - method, trigger_text, confidence, has_uncertainty, rejection_reason
    """
    if not arg1_text or not arg2_text or arg1_text.strip().lower() == arg2_text.strip().lower():
        return {"is_causal": 0, "pred_directed": 0, "rejection_reason": "self_loop"}
    if between_text is None or pd.isna(between_text) or str(between_text).strip() == "":
        return {"is_causal": 0, "pred_directed": 0, "rejection_reason": "empty_evidence_text"}
    
    between_text = str(between_text)
    if len(between_text) > max_between_chars:
        return {"is_causal": 0, "pred_directed": 0, "rejection_reason": "distance_window_exceeded"}
        
    if CORRELATION_PATTERN.search(between_text) and not any(p[1].search(between_text) for p in ACTIVE_CAUSAL_PATTERNS):
        return {"is_causal": 0, "pred_directed": 0, "rejection_reason": "correlation_not_causation"}
        
    if NEGATION_PATTERN.search(between_text):
        return {"is_causal": 0, "pred_directed": 0, "is_negated": True, "rejection_reason": "negation_detected"}
        
    has_uncertainty = bool(UNCERTAINTY_PATTERN.search(between_text))
    confidence = 0.70 if has_uncertainty else 0.90
    
    # Active patterns: syntactic ordering is (Cause) [verb] (Effect)
    for pname, pat in ACTIVE_CAUSAL_PATTERNS + DISCOURSE_PATTERNS:
        match = pat.search(between_text)
        if match:
            if arg1_before_arg2:
                # arg1 is before arg2 in text: arg1 -> arg2
                return {
                    "is_causal": 1, "pred_directed": 1, "cause": arg1_text, "effect": arg2_text,
                    "direction": "arg1->arg2", "method": "lexical_active",
                    "pattern_name": pname, "trigger_text": match.group(0),
                    "confidence": confidence, "has_uncertainty": has_uncertainty
                }
            else:
                # arg2 is before arg1 in text: arg2 -> arg1
                return {
                    "is_causal": 1, "pred_directed": 0, "cause": arg2_text, "effect": arg1_text,
                    "direction": "arg2->arg1", "method": "lexical_active",
                    "pattern_name": pname + "_REVERSED", "trigger_text": match.group(0),
                    "confidence": confidence, "has_uncertainty": has_uncertainty
                }
            
    # Passive patterns: syntactic ordering is (Effect) [passive verb] (Cause)
    for pname, pat in PASSIVE_CAUSAL_PATTERNS:
        match = pat.search(between_text)
        if match:
            if arg1_before_arg2:
                # arg1 is effect, arg2 is cause: arg2 -> arg1
                return {
                    "is_causal": 1, "pred_directed": 0, "cause": arg2_text, "effect": arg1_text,
                    "direction": "arg2->arg1", "method": "lexical_passive",
                    "pattern_name": pname + "_INVERTED", "trigger_text": match.group(0),
                    "confidence": confidence, "has_uncertainty": has_uncertainty
                }
            else:
                # arg2 is effect, arg1 is cause: arg1 -> arg2
                return {
                    "is_causal": 1, "pred_directed": 1, "cause": arg1_text, "effect": arg2_text,
                    "direction": "arg1->arg2", "method": "lexical_passive",
                    "pattern_name": pname, "trigger_text": match.group(0),
                    "confidence": confidence, "has_uncertainty": has_uncertainty
                }
            
    return {"is_causal": 0, "pred_directed": 0, "rejection_reason": "no_pattern_matched"}


def extract_dependency_relation(
    arg1_text: str,
    arg2_text: str,
    between_text: str,
    arg1_before_arg2: int = 1,
    max_between_chars: int = 400,
    nlp=None
) -> Dict[str, Any]:
    """Extract causal relation using spaCy syntactic dependency parse trees.
    Determines grammatical head predicate, active vs passive voice, and directional correctness.
    """
    if not arg1_text or not arg2_text or arg1_text.strip().lower() == arg2_text.strip().lower():
        return {"is_causal": 0, "pred_directed": 0, "rejection_reason": "self_loop"}
    if between_text is None or pd.isna(between_text) or str(between_text).strip() == "":
        return {"is_causal": 0, "pred_directed": 0, "rejection_reason": "empty_evidence_text"}
        
    between_text = str(between_text)
    if len(between_text) > max_between_chars:
        return {"is_causal": 0, "pred_directed": 0, "rejection_reason": "distance_window_exceeded"}
        
    if nlp is None:
        nlp = get_spacy_nlp()
        
    text = f"{arg1_text} {between_text} {arg2_text}" if arg1_before_arg2 else f"{arg2_text} {between_text} {arg1_text}"
    doc = nlp(text[:300])
    
    # Check negation
    if any(tok.dep_ == "neg" or tok.lower_ in NEG_CUES for tok in doc):
        return {"is_causal": 0, "pred_directed": 0, "is_negated": True, "rejection_reason": "negation_detected"}
        
    # Check correlation control
    if any(tok.lemma_ in {"correlate", "associate"} for tok in doc) and not any(tok.lemma_ in CAUSAL_LEMMAS for tok in doc):
        return {"is_causal": 0, "pred_directed": 0, "rejection_reason": "correlation_not_causation"}
        
    has_uncertainty = any(tok.lemma_ in {"may", "might", "could", "suggest", "possible"} for tok in doc)
    confidence = 0.70 if has_uncertainty else 0.90
    
    for token in doc:
        if token.lemma_ in CAUSAL_LEMMAS:
            # Check passive/inverted syntax (results from, is caused by, is mediated by)
            is_inverted = False
            if token.lemma_ == "result" and any(c.lower_ == "from" for c in token.children):
                is_inverted = True
            elif any(c.dep_ == "agent" or c.lower_ == "by" for c in token.children) or any(c.dep_ == "nsubjpass" for c in token.children):
                is_inverted = True
                
            if is_inverted:
                # Text order is Effect [verb] Cause
                if arg1_before_arg2:
                    # arg1 is first (Effect), arg2 is second (Cause) -> relation is arg2 -> arg1
                    return {
                        "is_causal": 1, "pred_directed": 0, "cause": arg2_text, "effect": arg1_text,
                        "direction": "arg2->arg1", "method": "dependency_passive",
                        "trigger_text": token.text, "confidence": confidence,
                        "has_uncertainty": has_uncertainty
                    }
                else:
                    # arg2 is first (Effect), arg1 is second (Cause) -> relation is arg1 -> arg2
                    return {
                        "is_causal": 1, "pred_directed": 1, "cause": arg1_text, "effect": arg2_text,
                        "direction": "arg1->arg2", "method": "dependency_passive",
                        "trigger_text": token.text, "confidence": confidence,
                        "has_uncertainty": has_uncertainty
                    }
            else:
                # Active order: Cause [verb] Effect
                if arg1_before_arg2:
                    # arg1 is first (Cause), arg2 is second (Effect) -> relation is arg1 -> arg2
                    return {
                        "is_causal": 1, "pred_directed": 1, "cause": arg1_text, "effect": arg2_text,
                        "direction": "arg1->arg2", "method": "dependency_active",
                        "trigger_text": token.text, "confidence": confidence,
                        "has_uncertainty": has_uncertainty
                    }
                else:
                    # arg2 is first (Cause), arg1 is second (Effect) -> relation is arg2 -> arg1
                    return {
                        "is_causal": 1, "pred_directed": 0, "cause": arg2_text, "effect": arg1_text,
                        "direction": "arg2->arg1", "method": "dependency_active",
                        "trigger_text": token.text, "confidence": confidence,
                        "has_uncertainty": has_uncertainty
                    }
                
    return {"is_causal": 0, "pred_directed": 0, "rejection_reason": "no_causal_lemma_found"}


def extract_hybrid_relation(
    arg1_text: str,
    arg2_text: str,
    between_text: str,
    arg1_before_arg2: int = 1,
    max_between_chars: int = 400,
    nlp=None
) -> Dict[str, Any]:
    """Hybrid combination: predicts directed causal if either lexical rules or dependency parsing detects directed causality."""
    r_lex = extract_lexical_relation(arg1_text, arg2_text, between_text, arg1_before_arg2, max_between_chars)
    if r_lex["pred_directed"] == 1:
        return r_lex
    r_dep = extract_dependency_relation(arg1_text, arg2_text, between_text, arg1_before_arg2, max_between_chars, nlp)
    if r_dep["pred_directed"] == 1:
        return r_dep
    return r_lex


def build_extracted_records(
    df_cands: pd.DataFrame,
    extractor_fn,
    method_id: str,
    method_name: str,
    schema_columns: List[str]
) -> pd.DataFrame:
    """Standardized batch transformation from candidate pairs to validated UnifiedRelationRecords."""
    records = []
    for _, r in df_cands.iterrows():
        res = extractor_fn(
            arg1_text=str(r["arg1_text"]),
            arg2_text=str(r["arg2_text"]),
            between_text=str(r["between_text"]),
            arg1_before_arg2=int(r["arg1_before_arg2"])
        )
        if res["pred_directed"] == 1:
            records.append({
                "relation_id": f"{method_id}_{r['candidate_id']}",
                "article_id": r["article_id"],
                "section_id": r["section_id"],
                "cause_id": r["arg1_id"],
                "cause_text": r["arg1_text"],
                "cause_start": r["arg1_start"],
                "cause_end": r["arg1_end"],
                "effect_id": r["arg2_id"],
                "effect_text": r["arg2_text"],
                "effect_start": r["arg2_start"],
                "effect_end": r["arg2_end"],
                "direction": "cause->effect",
                "confidence": float(res.get("confidence", 0.90)),
                "extraction_method": method_name,
                "evidence_text": str(r["between_text"]),
                "provenance": str(res.get("trigger_text", res.get("method", method_name))),
                "split": r["split"]
            })
    return pd.DataFrame(records, columns=schema_columns)
