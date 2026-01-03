import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("ground_truth", help="Path to the ground truth file")
parser.add_argument("predictions", help="Path to the predictions file")
args = parser.parse_args()

def recall_at_k(pred_list, gold_list, k):
    if not pred_list:
        return 0.0
    # Function to calculate recall at k
    pred_set = set(pred_list[:k])
    gold_set = set(gold_list)
    return len(pred_set.intersection(gold_set)) / len(gold_set) if gold_set else 0

def case_f1_score(pred_list, gold_list):
    pred_set = set(pred_list)
    gold_set = set(gold_list)
    if not pred_set:
        return 0.0
    precision = len(pred_set.intersection(gold_set)) / len(pred_set)
    recall = len(pred_set.intersection(gold_set)) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evidence_score(pred_list, gold_list, k=5, r_at_k_threshold=0.5):
    r_at_k = recall_at_k(pred_list, gold_list, k)
    if r_at_k < r_at_k_threshold:
        return 0.0
    return case_f1_score(pred_list, gold_list)

def main():
    ground_truth = pd.read_json(args.ground_truth, lines=True)
    predictions = pd.read_json(args.predictions, lines=True)

    predictions = ground_truth.merge(predictions, on="claim", how="left")

    # The predicted evidence must first pass a R@5 filter, 
    # then the evidence score will be calculated as:
    #   evidence score = F1(true_cases, pred_cases)
    # then the verdict score will be computed as:
    #   evidence score * verdict accuracy

    predictions['evidence_score'] = predictions.apply(lambda row: evidence_score(row['predicted_cases'], row['case_names'], k=5), axis=1)

    predictions['verdict_accuracy'] = (predictions['predicted_verdict'] == predictions['verdict']).astype(float)

    predictions['verdict_score'] = predictions['evidence_score'] * predictions['verdict_accuracy']

    print(f"Average Evidence Score: {predictions['evidence_score'].mean():.3f}")
    print(f"Average Verdict Accuracy: {predictions['verdict_accuracy'].mean():.3f}")
    print(f"Average Verdict Score: {predictions['verdict_score'].mean():.3f}")

if __name__ == "__main__":
    main()