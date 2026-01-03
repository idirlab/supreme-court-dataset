#!/usr/bin/env python3
"""
eval_legal_citations_ratk_main.py

Evaluation script implementing the agreed "R@K filter then (case_F1 * verdict_correct)" main metric,
plus other metrics (exact match, F1-threshold gate, P@K/R@K, micro case precision/recall/F1).

Usage example:
 python eval_legal_citations_ratk_main.py --gold test_set.csv --pred preds.csv \
    --pred_cases_col predicted_cases --pred_label_col predicted_verdict \
    --k 5 --rk_threshold 0.5

Default behavior:
 - Main metric: R@K filter (default K=5, rk_threshold=0.5). If passed, per-sample score = case_F1 * verdict_correct (0 or 1).
 - Dataset main metric = mean per-sample score.
"""

from __future__ import annotations
import argparse
import ast
import json
import re
from typing import List, Set, Tuple, Dict, Optional

import pandas as pd

# ---------- Parsing & Normalization ----------

def parse_case_list(raw) -> List[str]:
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
    if ";" in s:
        return [p.strip() for p in s.split(";") if p.strip() != ""]
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip() != ""]
    if '","' in s:
        return [p.strip().strip('"').strip("'") for p in s.split('","') if p.strip() != ""]
    if "," in s:
        return [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip() != ""]
    return [s]

def normalize_case_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).lower().strip()
    s = re.sub(r'\bvs?\.?(\s+|$)', ' v ', s)  # normalize vs./v.
    s = re.sub(r'[^0-9a-z v]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def list_to_norm_set(lst: List[str]) -> Set[str]:
    return {normalize_case_name(x) for x in lst if x is not None and str(x).strip() != ""}

# ---------- Basic set metrics ----------

def precision_recall_f1_for_sets(pred_set: Set[str], gold_set: Set[str]) -> Tuple[float, float, float, int]:
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0, 0
    tp = len(pred_set.intersection(gold_set))
    prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    rec = tp / len(gold_set) if len(gold_set) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp

# ---------- P@K and R@K helpers ----------

def prec_at_k(pred_list: List[str], gold_set: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    if (not gold_set) and (not pred_list):
        return 1.0
    topk = pred_list[:k]
    topk_norm = [normalize_case_name(x) for x in topk]
    tp = sum(1 for x in topk_norm if x in gold_set)
    denom = min(k, len(pred_list)) if len(pred_list) > 0 else k
    return tp / denom if denom > 0 else 0.0

def rec_at_k(pred_list: List[str], gold_set: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    if not gold_set:
        return 1.0
    topk = pred_list[:k]
    topk_norm = [normalize_case_name(x) for x in topk]
    tp = sum(1 for x in topk_norm if x in gold_set)
    return tp / len(gold_set)

# ---------- Main new metric: R@K filter then case_F1 * verdict_correct ----------

def main_metric_ratk_filter_mul(
    pred_cases: List[str],
    pred_label: str,
    gold_cases: List[str],
    gold_label: str,
    gold_overruling_case: Optional[str] = None,
    k: int = 5,
    rk_threshold: float = 0.5,
    enforce_overruling_for_overruled: bool = True
) -> Dict:
    """
    Main metric:
      - compute R@K
      - graduation if R@K >= rk_threshold, and (if Overruled) overruling_case present (configurable)
      - if graduated: sample_score = case_F1 * verdict_correct (verdict_correct in {0,1})
      - else: sample_score = 0
    Returns details for reporting.
    """
    pset = list_to_norm_set(pred_cases)
    gset = list_to_norm_set(gold_cases)
    prec, rec, f1, tp = precision_recall_f1_for_sets(pset, gset)
    r_at_k = rec_at_k(pred_cases, gset, k)
    p_at_k = prec_at_k(pred_cases, gset, k)

    # Graduation by R@K
    graduated = r_at_k >= rk_threshold

    # enforce overruling presence when configured
    if gold_label and str(gold_label).strip().lower() == "overruled" and enforce_overruling_for_overruled:
        if gold_overruling_case:
            norm_o = normalize_case_name(gold_overruling_case)
            if norm_o not in pset:
                graduated = False

    verdict_correct = 1 if str(pred_label).strip().lower() == str(gold_label).strip().lower() else 0

    sample_score = f1 * verdict_correct if graduated else 0.0

    return {
        "precision_full": prec,
        "recall_full": rec,
        "f1_full": f1,
        "tp_count": tp,
        "p_at_k": p_at_k,
        "r_at_k": r_at_k,
        "graduated": graduated,
        "verdict_correct": bool(verdict_correct),
        "sample_score": sample_score
    }

# ---------- Existing metrics (exact match, F1-threshold & flexible metric) ----------

def main_metric_with_thresholds(
    pred_cases: List[str],
    pred_label: str,
    gold_cases: List[str],
    gold_label: str,
    gold_overruling_case: Optional[str] = None,
    f1_threshold: Optional[float] = 0.8,
    pk_threshold: Optional[float] = None,
    rk_threshold: Optional[float] = None,
    k: int = 5,
    enforce_overruling_for_overruled: bool = True
) -> Dict:
    pset = list_to_norm_set(pred_cases)
    gset = list_to_norm_set(gold_cases)
    prec, rec, f1, tp = precision_recall_f1_for_sets(pset, gset)
    p_at_k = prec_at_k(pred_cases, gset, k)
    r_at_k = rec_at_k(pred_cases, gset, k)

    graduated = False
    if f1_threshold is not None and f1 >= f1_threshold:
        graduated = True
    if (pk_threshold is not None) and p_at_k >= pk_threshold:
        graduated = True
    if (rk_threshold is not None) and r_at_k >= rk_threshold:
        graduated = True

    if gold_label and str(gold_label).strip().lower() == "overruled" and enforce_overruling_for_overruled:
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
        "p_at_k": p_at_k,
        "r_at_k": r_at_k,
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
    pset = list_to_norm_set(pred_cases)
    gset = list_to_norm_set(gold_cases)
    exact_match = (pset == gset)
    if gold_label and str(gold_label).strip().lower() == "overruled" and require_overruling_in_exact:
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

# ---------- Aggregated cases metrics (with P@K/R@K) ----------

def cases_precision_recall_f1_with_pk_rk(
    all_pred_cases_lists: List[List[str]],
    all_gold_cases_lists: List[List[str]],
    k: int = 5
) -> Dict:
    total_tp = 0
    total_pred = 0
    total_gold = 0
    f1s = []
    p_at_ks = []
    r_at_ks = []
    for pred, gold in zip(all_pred_cases_lists, all_gold_cases_lists):
        pset = list_to_norm_set(pred)
        gset = list_to_norm_set(gold)
        prec, rec, f1, tp = precision_recall_f1_for_sets(pset, gset)
        total_tp += tp
        total_pred += len(pset)
        total_gold += len(gset)
        f1s.append(f1)
        p_at_ks.append(prec_at_k(pred, gset, k))
        r_at_ks.append(rec_at_k(pred, gset, k))

    micro_prec = total_tp / total_pred if total_pred > 0 else 0.0
    micro_rec = total_tp / total_gold if total_gold > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0
    mean_per_sample_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    mean_p_at_k = sum(p_at_ks) / len(p_at_ks) if p_at_ks else 0.0
    mean_r_at_k = sum(r_at_ks) / len(r_at_ks) if r_at_ks else 0.0

    total_tp_at_k = 0
    total_denom_for_pk = 0
    total_gold_for_rk = 0
    for pred, gold in zip(all_pred_cases_lists, all_gold_cases_lists):
        topk = pred[:k]
        topk_norm = [normalize_case_name(x) for x in topk]
        gset = list_to_norm_set(gold)
        tp_k = sum(1 for x in topk_norm if x in gset)
        total_tp_at_k += tp_k
        denom_pk = min(k, len(pred)) if len(pred) > 0 else k
        total_denom_for_pk += denom_pk
        total_gold_for_rk += len(gset)

    micro_p_at_k = total_tp_at_k / total_denom_for_pk if total_denom_for_pk > 0 else 0.0
    micro_r_at_k = total_tp_at_k / total_gold_for_rk if total_gold_for_rk > 0 else 0.0

    return {
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "mean_per_sample_f1": mean_per_sample_f1,
        "mean_p_at_k": mean_p_at_k,
        "mean_r_at_k": mean_r_at_k,
        "micro_p_at_k": micro_p_at_k,
        "micro_r_at_k": micro_r_at_k,
        "total_tp": total_tp,
        "total_tp_at_k": total_tp_at_k,
        "total_pred": total_pred,
        "total_gold": total_gold
    }

# ---------- File orchestration ----------

def evaluate_files(
    gold_path: str,
    pred_path: str,
    claim_key: str = "claim",
    gold_cases_col: str = "case_name",
    gold_label_col: str = "label",
    gold_overrule_col: str = "overruling_case",
    pred_cases_col: str = "predicted_cases",
    pred_label_col: str = "predicted_verdict",
    # defaults for various thresholds
    main_k: int = 5,
    main_rk_threshold: float = 0.5,
    f1_threshold: Optional[float] = 0.8,
    pk_threshold: Optional[float] = None,
    rk_threshold: Optional[float] = None,  # additional flexible thresholds if used
    enforce_overruling_for_overruled: bool = True,
    require_overruling_in_exact: bool = True,
    output_path: str = "eval_results.csv"
) -> Dict:
    gold = pd.read_csv(gold_path)
    preds = pd.read_csv(pred_path)

    merged = gold.merge(preds, on=claim_key, how='left', suffixes=('_gold', '_pred'))

    parsed_pred_cases = []
    parsed_gold_cases = []
    gold_overruling = []
    pred_labels = []
    gold_labels = []
    per_sample_results = []

    for idx, row in merged.iterrows():
        claim_text = row.get(claim_key, "")
        raw_gold_cases = row.get(gold_cases_col, "")
        gold_cases_list = parse_case_list(raw_gold_cases)
        raw_overrule = row.get(gold_overrule_col, None) if gold_overrule_col in row.index else None
        gold_label = row.get(gold_label_col, "")

        raw_pred_cases = row.get(pred_cases_col, "")
        if pd.isna(raw_pred_cases):
            pred_cases_list = []
        else:
            pred_cases_list = parse_case_list(raw_pred_cases)
        pred_label = row.get(pred_label_col, "") if pred_label_col in row.index else ""

        parsed_gold_cases.append(gold_cases_list)
        parsed_pred_cases.append(pred_cases_list)

        # --- main (new) metric ---
        main_ratk = main_metric_ratk_filter_mul(
            pred_cases_list, pred_label, gold_cases_list, gold_label,
            gold_overruling_case=raw_overrule,
            k=main_k,
            rk_threshold=main_rk_threshold,
            enforce_overruling_for_overruled=enforce_overruling_for_overruled
        )

        # --- other metrics for reporting ---
        flexible = main_metric_with_thresholds(
            pred_cases_list, pred_label, gold_cases_list, gold_label,
            gold_overruling_case=raw_overrule,
            f1_threshold=f1_threshold,
            pk_threshold=pk_threshold,
            rk_threshold=rk_threshold,
            k=main_k,
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
            # main ratk metric fields
            "main_ratk_p_at_k": main_ratk["p_at_k"],
            "main_ratk_r_at_k": main_ratk["r_at_k"],
            "main_ratk_f1_full": main_ratk["f1_full"],
            "main_ratk_graduated": main_ratk["graduated"],
            "main_ratk_verdict_correct": main_ratk["verdict_correct"],
            "main_ratk_sample_score": main_ratk["sample_score"],
            # flexible metric fields
            "flex_prec": flexible["precision"],
            "flex_rec": flexible["recall"],
            "flex_f1": flexible["f1"],
            "flex_graduated": flexible["graduated"],
            "flex_correct": flexible["correct"],
            # exact metrics
            "exact_match": exact_res["exact_match"],
            "exact_correct": exact_res["correct"],
            "verdict_only_correct": verdict_only
        })

    # Aggregations
    n_total = len(per_sample_results)
    mean_main_ratk_score = sum(r["main_ratk_sample_score"] for r in per_sample_results) / n_total if n_total > 0 else 0.0

    # Bin-count: how many samples graduated by the R@K filter (for inspection)
    n_main_ratk_graduated = sum(1 for r in per_sample_results if r["main_ratk_graduated"])
    frac_main_ratk_graduated = n_main_ratk_graduated / n_total if n_total > 0 else 0.0

    # other metrics (flexible, exact, verdict-only)
    n_flex_correct = sum(1 for r in per_sample_results if r["flex_correct"])
    flex_accuracy = n_flex_correct / n_total if n_total > 0 else 0.0

    n_exact_correct = sum(1 for r in per_sample_results if r["exact_correct"])
    exact_accuracy = n_exact_correct / n_total if n_total > 0 else 0.0

    n_verdict_correct = sum(1 for r in per_sample_results if r["verdict_only_correct"])
    verdict_accuracy = n_verdict_correct / n_total if n_total > 0 else 0.0

    cases_stats = cases_precision_recall_f1_with_pk_rk(parsed_pred_cases, parsed_gold_cases, k=main_k)

    out_df = pd.DataFrame(per_sample_results)
    out_df.to_csv(output_path, index=False)

    summary = {
        "n_samples": n_total,
        # MAIN metric (R@K filter then case_F1 * verdict)
        "main_metric_name": "R@K-filter × (case_F1 × verdict_correct)",
        "main_k": main_k,
        "k": main_k,
        "main_rk_threshold": main_rk_threshold,
        "main_mean_score": mean_main_ratk_score,
        "main_frac_graduated": frac_main_ratk_graduated,
        # flexible metric (legacy)
        "flexible_metric_accuracy": flex_accuracy,
        "f1_threshold": f1_threshold,
        "pk_threshold": pk_threshold,
        "rk_threshold": rk_threshold,
        # exact-match
        "exact_accuracy": exact_accuracy,
        # verdict-only
        "verdict_only_accuracy": verdict_accuracy,
        # cases micro stats
        "cases_micro_precision": cases_stats["micro_precision"],
        "cases_micro_recall": cases_stats["micro_recall"],
        "cases_micro_f1": cases_stats["micro_f1"],
        "cases_mean_per_sample_f1": cases_stats["mean_per_sample_f1"],
        "cases_mean_p_at_k": cases_stats["mean_p_at_k"],
        "cases_mean_r_at_k": cases_stats["mean_r_at_k"],
        "cases_micro_p_at_k": cases_stats["micro_p_at_k"],
        "cases_micro_r_at_k": cases_stats["micro_r_at_k"],
        "per_sample_output": output_path
    }

    return {"summary": summary, "per_sample_df": out_df}

# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate legal citation + verdict predictions with R@K-filter main metric.")
    parser.add_argument("--gold", default="test_set.csv", help="Gold test set CSV (e.g., test_set.csv)")
    parser.add_argument("--pred", required=True, help="Predictions CSV")
    parser.add_argument("--claim_col", default="claim", help="Column name for claim text (used to join files)")
    parser.add_argument("--gold_cases_col", default="case_name", help="Gold column with cases")
    parser.add_argument("--gold_label_col", default="label", help="Gold column with verdict/label")
    parser.add_argument("--gold_overrule_col", default="overruling_case", help="Gold column with overruling case (optional)")
    parser.add_argument("--pred_cases_col", default="predicted_cases", help="Predictions column with predicted cases")
    parser.add_argument("--pred_label_col", default="predicted_verdict", help="Predictions column with predicted verdict")
    parser.add_argument("--k", type=int, default=5, help="K for P@K / R@K (default 5)")
    parser.add_argument("--rk_threshold", type=float, default=0.75, help="R@K threshold used as the graduation filter for the main metric (default 0.5)")
    parser.add_argument("--threshold", type=float, default=0.8, help="F1 threshold for flexible metric graduation (default 0.8)")
    parser.add_argument("--pk_threshold", type=float, default=None, help="Precision@K threshold for flexible graduation (optional)")
    parser.add_argument("--rk_threshold_flex", type=float, default=None, help="Recall@K threshold for flexible graduation (optional; separate from main R@K filter)")
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
        main_k=args.k,
        main_rk_threshold=args.rk_threshold,
        f1_threshold=args.threshold,
        pk_threshold=args.pk_threshold,
        rk_threshold=args.rk_threshold_flex,
        enforce_overruling_for_overruled=not args.no_enforce_overruling,
        require_overruling_in_exact=not args.no_require_overruling_in_exact,
        output_path=args.out
    )
    s = res["summary"]
    print("=== Evaluation Summary ===")
    print(f"Samples: {s['n_samples']}")
    print("MAIN METRIC (printed first):")
    print(f"  Name: {s['main_metric_name']}")
    print(f"  K = {s['main_k']}, R@K threshold = {s['main_rk_threshold']}")
    print(f"  Mean per-sample main score = {s['main_mean_score']:.4f}   (range 0..1)")
    print(f"  Fraction of samples that graduated R@K filter = {s['main_frac_graduated']:.4f}")
    print("")
    print("Other metrics:")
    print(f"  Flexible (F1/P@K/R@K) metric accuracy = {s['flexible_metric_accuracy']:.4f}")
    print(f"  Exact-match (cases sets identical) accuracy = {s['exact_accuracy']:.4f}")
    print(f"  Verdict-only accuracy = {s['verdict_only_accuracy']:.4f}")
    print("  Cases (micro P / R / F1) = "
          f"{s['cases_micro_precision']:.4f} / {s['cases_micro_recall']:.4f} / {s['cases_micro_f1']:.4f}")
    print(f"  Mean per-sample case F1 = {s['cases_mean_per_sample_f1']:.4f}")
    print(f"  Mean per-sample P@{s['k']} = {s['cases_mean_p_at_k']:.4f}")
    print(f"  Mean per-sample R@{s['k']} = {s['cases_mean_r_at_k']:.4f}")
    print(f"  Micro P@{s['k']} = {s['cases_micro_p_at_k']:.4f}")
    print(f"  Micro R@{s['k']} = {s['cases_micro_r_at_k']:.4f}")
    print(f"Per-sample results written to: {s['per_sample_output']}")
    print("==========================")

if __name__ == "__main__":
    main()
