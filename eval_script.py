"""
Evaluation script for legal claim + case prediction tasks.

Main metric (default):
 - For each claim compute case-level precision/recall/F1 between predicted_cases and gold case_name.
 - If F1 >= threshold (default 0.8) the sample "graduates".
 - If graduated AND predicted_verdict == gold verdict, mark as correct.
 - Overall metric: accuracy = (# correct) / (# evaluated samples)

Other metrics implemented:
 - exact_case_match_metric: requires predicted case set == gold case set (optionally require overruling presence)
 - verdict_accuracy_only: label-only accuracy
 - cases_precision_recall_f1: micro precision/recall/F1 across all predictions, and mean per-sample F1

Outputs:
 - Printed summary
 - Per-sample CSV: eval_results.csv

Assumptions / Notes:
 - Gold CSV should be in the format you provided: columns include at least:
     - claim
     - case_name (a JSON list-like string or other list format)
     - overruling_case (optional; only used if enforce_overruling flags are enabled)
     - label (gold verdict string)
 - Predictions CSV should include:
     - claim (to match to gold)
     - predicted_cases (string; robust parsing supported)
     - predicted_verdict
 - Matching between gold and predicted cases uses a lightweight normalization to reduce surface-form mismatches.
"""

from __future__ import annotations
import argparse
import ast
import csv
import json
import math
import os
import re
from typing import List, Set, Tuple, Dict, Optional

import pandas as pd

# ---------- Utilities: parsing & normalization ----------

def parse_case_list(raw) -> List[str]:
    """
    Best-effort parse a predicted or gold 'cases' field.
    Accepts:
      - JSON array strings: '["A v. B", "C v. D"]'
      - Python list literals: "['A v. B', 'C v. D']"
      - Semicolon- or pipe-separated strings: "A v. B; C v. D" or "A v. B | C v. D"
      - Comma-separated if no JSON-like structure (fallback)
      - If already a list, returns it
      - None or empty -> []
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None and str(x).strip() != ""]
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none", "[]"}:
        return []
    # Try JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x is not None and str(x).strip() != ""]
    except Exception:
        pass
    # Try Python literal
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed if x is not None and str(x).strip() != ""]
    except Exception:
        pass
    # Split by common separators
    # Prefer semicolon or pipe as safe separators; comma is ambiguous inside names but use as last resort
    if ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip() != ""]
        return parts
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip() != ""]
        return parts
    # If the string contains '","' sequence (raw JSON-ish without outer []), try splitting on '","'
    if '","' in s:
        parts = [p.strip().strip('"').strip("'") for p in s.split('","') if p.strip() != ""]
        return parts
    # fallback: split on comma but be conservative
    if "," in s:
        parts = [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip() != ""]
        # If parts look like single long names incorrectly split, try to detect (heuristic) - but we keep simple.
        return parts
    # Single case name
    return [s]

def normalize_case_name(name: str) -> str:
    """
    Normalize case names to reduce superficial mismatches:
    - lowercases
    - strips leading/trailing whitespace
    - removes punctuation except letters, numbers, spaces and the letter 'v' which often appears in 'v.' or 'vs'
    - collapses multiple spaces
    This is conservative: it's intended to reduce formatting mismatches but not to conflate distinct cases.
    """
    if name is None:
        return ""
    s = str(name).lower().strip()
    # replace common variants of 'v.' and 'vs.' to 'v'
    s = re.sub(r'\bvs?\.?(\s+|$)', ' v ', s)
    # remove punctuation except alphanumerics, spaces and 'v'
    s = re.sub(r'[^0-9a-z v]', ' ', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def list_to_norm_set(lst: List[str]) -> Set[str]:
    return {normalize_case_name(x) for x in lst if x is not None and str(x).strip() != ""}

# ---------- Core metrics ----------

def precision_recall_f1_for_sets(pred_set: Set[str], gold_set: Set[str]) -> Tuple[float, float, float, int]:
    """
    Return (precision, recall, f1, tp_count) for the two sets.
    If both sets are empty: define precision=recall=f1=1.0 (perfect match).
    If pred empty and gold nonempty: precision=0.0, recall=0.0, f1=0.0.
    """
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0, 0
    tp = len(pred_set.intersection(gold_set))
    prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    rec = tp / len(gold_set) if len(gold_set) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp

# ---------- Evaluator functions ----------

def main_metric_f1_threshold(
    pred_cases: List[str],
    pred_label: str,
    gold_cases: List[str],
    gold_label: str,
    gold_overruling_case: Optional[str] = None,
    threshold: float = 0.8,
    enforce_overruling_for_overruled: bool = True
) -> Dict:
    """
    Implements your requested main metric:
      - compute case F1 between predicted and gold sets
      - 'graduate' if F1 >= threshold
      - OPTIONAL: if gold_label == 'Overruled' and enforce_overruling_for_overruled True,
                  require normalized overruling_case be present in predictions to graduate
      - if graduated AND pred_label == gold_label => correct
    Returns a dict with fields:
      - precision, recall, f1, tp_count
      - graduated (bool)
      - correct (bool)
    """
    pset = list_to_norm_set(pred_cases)
    gset = list_to_norm_set(gold_cases)
    prec, rec, f1, tp = precision_recall_f1_for_sets(pset, gset)

    graduated = f1 >= threshold
    # check overruling presence if required
    if gold_label and gold_label.lower() == "overruled" and enforce_overruling_for_overruled:
        if gold_overruling_case:
            norm_o = normalize_case_name(gold_overruling_case)
            if norm_o not in pset:
                graduated = False

    correct = graduated and str(pred_label).strip().lower() == str(gold_label).strip().lower()
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp_count": tp,
        "graduated": graduated,
        "correct": correct
    }

def exact_case_match_metric(
    pred_cases: List[str],
    pred_label: str,
    gold_cases: List[str],
    gold_label: str,
    gold_overruling_case: Optional[str] = None,
    require_overruling_in_exact: bool = True
) -> Dict:
    """
    Exact-match variant: the gate requires predicted case set EXACTLY equals gold case set (no more, no less).
    For Overruled gold labels we optionally require that the gold_overruling_case be included in the predicted set.
    """
    pset = list_to_norm_set(pred_cases)
    gset = list_to_norm_set(gold_cases)
    exact_match = (pset == gset)
    if gold_label and gold_label.lower() == "overruled" and require_overruling_in_exact:
        if gold_overruling_case:
            norm_o = normalize_case_name(gold_overruling_case)
            if norm_o not in pset:
                exact_match = False
    correct = exact_match and str(pred_label).strip().lower() == str(gold_label).strip().lower()
    return {
        "exact_match": exact_match,
        "correct": correct,
        "pred_set_size": len(pset),
        "gold_set_size": len(gset)
    }

def verdict_accuracy_only(pred_label: str, gold_label: str) -> bool:
    return str(pred_label).strip().lower() == str(gold_label).strip().lower()

def cases_precision_recall_f1(
    all_pred_cases_lists: List[List[str]],
    all_gold_cases_lists: List[List[str]]
) -> Dict:
    """
    Compute micro-averaged precision/recall/f1 across all items (sum TP / sum Pred / sum Gold),
    and return mean per-sample F1 as well (macro-like mean of per-sample F1).
    """
    total_tp = 0
    total_pred = 0
    total_gold = 0
    f1s = []
    for pred, gold in zip(all_pred_cases_lists, all_gold_cases_lists):
        pset = list_to_norm_set(pred)
        gset = list_to_norm_set(gold)
        prec, rec, f1, tp = precision_recall_f1_for_sets(pset, gset)
        total_tp += tp
        total_pred += len(pset)
        total_gold += len(gset)
        f1s.append(f1)
    micro_prec = total_tp / total_pred if total_pred > 0 else 0.0
    micro_rec = total_tp / total_gold if total_gold > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0
    mean_per_sample_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "mean_per_sample_f1": mean_per_sample_f1,
        "total_tp": total_tp,
        "total_pred": total_pred,
        "total_gold": total_gold
    }

# ---------- Orchestration: process CSVs and compute metrics ----------

def evaluate_files(
    gold_path: str,
    pred_path: str,
    claim_key: str = "claim",
    gold_cases_col: str = "case_name",
    gold_label_col: str = "label",
    gold_overrule_col: str = "overruling_case",
    pred_cases_col: str = "predicted_cases",
    pred_label_col: str = "predicted_verdict",
    f1_threshold: float = 0.8,
    enforce_overruling_for_overruled: bool = True,
    require_overruling_in_exact: bool = True,
    output_path: str = "eval_results.csv"
) -> Dict:
    # Load CSVs
    gold = pd.read_csv(gold_path)
    preds = pd.read_csv(pred_path)

    # Merge by claim (exact match)
    merged = gold.merge(preds, on=claim_key, how='left', suffixes=('_gold', '_pred'))
    # rows without prediction produce NaNs; we'll treat predicted as empty lists
    # Parse case lists and normalize into lists
    parsed_pred_cases = []
    parsed_gold_cases = []
    gold_overruling = []
    pred_labels = []
    gold_labels = []
    claims = []
    per_sample_results = []

    for idx, row in merged.iterrows():
        claim_text = row.get(claim_key, "")
        claims.append(claim_text)
        # gold
        raw_gold_cases = row.get(gold_cases_col, "")
        gold_cases_list = parse_case_list(raw_gold_cases)
        parsed_gold_cases.append(gold_cases_list)
        gold_label = row.get(gold_label_col, "")
        gold_labels.append(gold_label)
        raw_overrule = row.get(gold_overrule_col, None) if gold_overrule_col in row.index else None
        gold_overruling.append(raw_overrule)

        # preds (may be NaN)
        raw_pred_cases = row.get(pred_cases_col, "")
        if pd.isna(raw_pred_cases):
            pred_cases_list = []
        else:
            pred_cases_list = parse_case_list(raw_pred_cases)
        parsed_pred_cases.append(pred_cases_list)
        pred_label = row.get(pred_label_col, "") if pred_label_col in row.index else ""
        pred_labels.append(pred_label)

        # compute per-sample metrics
        main_res = main_metric_f1_threshold(
            pred_cases_list, pred_label, gold_cases_list, gold_label,
            gold_overruling_case=raw_overrule,
            threshold=f1_threshold,
            enforce_overruling_for_overruled=enforce_overruling_for_overruled
        )
        exact_res = exact_case_match_metric(
            pred_cases_list, pred_label, gold_cases_list, gold_label,
            gold_overruling_case=raw_overrule,
            require_overruling_in_exact=require_overruling_in_exact
        )
        verdict_only = verdict_accuracy_only(pred_label, gold_label)
        per_sample_results.append({
            "claim": claim_text,
            "predicted_cases_raw": raw_pred_cases,
            "predicted_cases_parsed": pred_cases_list,
            "gold_cases_parsed": gold_cases_list,
            "predicted_verdict": pred_label,
            "gold_verdict": gold_label,
            "gold_overruling_case": raw_overrule,
            "main_prec": main_res["precision"],
            "main_rec": main_res["recall"],
            "main_f1": main_res["f1"],
            "main_graduated": main_res["graduated"],
            "main_correct": main_res["correct"],
            "exact_match": exact_res["exact_match"],
            "exact_correct": exact_res["correct"],
            "verdict_only_correct": verdict_only
        })

    # compute dataset-level aggregations
    # main metric accuracy:
    n_total = len(per_sample_results)
    n_main_correct = sum(1 for r in per_sample_results if r["main_correct"])
    main_accuracy = n_main_correct / n_total if n_total > 0 else 0.0

    # exact-case metric accuracy:
    n_exact_correct = sum(1 for r in per_sample_results if r["exact_correct"])
    exact_accuracy = n_exact_correct / n_total if n_total > 0 else 0.0

    # verdict-only accuracy:
    n_verdict_correct = sum(1 for r in per_sample_results if r["verdict_only_correct"])
    verdict_accuracy = n_verdict_correct / n_total if n_total > 0 else 0.0

    # cases micro precision/recall/f1 and mean per-sample f1:
    cases_stats = cases_precision_recall_f1(parsed_pred_cases, parsed_gold_cases)

    # build data frame and save
    out_df = pd.DataFrame(per_sample_results)
    out_df.to_csv(output_path, index=False)

    summary = {
        "n_samples": n_total,
        "main_accuracy_f1_threshold": main_accuracy,
        "f1_threshold": f1_threshold,
        "enforce_overruling_for_overruled": enforce_overruling_for_overruled,
        "exact_accuracy": exact_accuracy,
        "require_overruling_in_exact": require_overruling_in_exact,
        "verdict_only_accuracy": verdict_accuracy,
        "cases_micro_precision": cases_stats["micro_precision"],
        "cases_micro_recall": cases_stats["micro_recall"],
        "cases_micro_f1": cases_stats["micro_f1"],
        "cases_mean_per_sample_f1": cases_stats["mean_per_sample_f1"],
        "per_sample_output": output_path
    }

    return {"summary": summary, "per_sample_df": out_df}

# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate legal citation + verdict predictions.")
    parser.add_argument("--gold", default="test_set.csv", help="Gold test set CSV (e.g., test_set.csv)")
    parser.add_argument("--pred", required=True, help="Predictions CSV")
    parser.add_argument("--claim_col", default="claim", help="Column name for claim text (used to join files)")
    parser.add_argument("--gold_cases_col", default="case_name", help="Gold column with cases")
    parser.add_argument("--gold_label_col", default="label", help="Gold column with verdict/label")
    parser.add_argument("--gold_overrule_col", default="overruling_case", help="Gold column with overruling case (optional)")
    parser.add_argument("--pred_cases_col", default="predicted_cases", help="Predictions column with predicted cases")
    parser.add_argument("--pred_label_col", default="predicted_verdict", help="Predictions column with predicted verdict")
    parser.add_argument("--threshold", type=float, default=0.8, help="F1 threshold for main metric graduation (default 0.8)")
    parser.add_argument("--no_enforce_overruling", action="store_true", help="Do not require overruling case presence for Overruled gold labels in main metric")
    parser.add_argument("--no_require_overruling_in_exact", action="store_true", help="Do not require overruling case presence for Overruled gold labels in exact-match metric")
    parser.add_argument("--out", default="eval_results.csv", help="Per-sample CSV output path")
    return parser.parse_args()

def main():
    args = parse_args()
    res = evaluate_files(
        gold_path=args.gold,
        pred_path=args.pred,
        claim_key=args.claim_col,
        gold_cases_col=args.gold_cases_col,
        gold_label_col=args.gold_label_col,
        gold_overrule_col=args.gold_overrule_col,
        pred_cases_col=args.pred_cases_col,
        pred_label_col=args.pred_label_col,
        f1_threshold=args.threshold,
        enforce_overruling_for_overruled=not args.no_enforce_overruling,
        require_overruling_in_exact=not args.no_require_overruling_in_exact,
        output_path=args.out
    )
    summary = res["summary"]
    print("=== Evaluation Summary ===")
    print(f"Samples: {summary['n_samples']}")
    print(f"Main metric (F1 threshold {summary['f1_threshold']}): accuracy = {summary['main_accuracy_f1_threshold']:.4f}")
    print(f"Exact-match (cases sets identical): accuracy = {summary['exact_accuracy']:.4f}")
    print(f"Verdict-only accuracy: {summary['verdict_only_accuracy']:.4f}")
    print(f"Cases micro precision / recall / f1 = {summary['cases_micro_precision']:.4f} / {summary['cases_micro_recall']:.4f} / {summary['cases_micro_f1']:.4f}")
    print(f"Mean per-sample case F1 = {summary['cases_mean_per_sample_f1']:.4f}")
    print(f"Per-sample results written to: {summary['per_sample_output']}")
    print("==========================")

if __name__ == "__main__":
    main()
