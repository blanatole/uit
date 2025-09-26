#!/usr/bin/env python3
"""
Script để đánh giá kết quả debate từ file CSV
"""
import pandas as pd
import argparse
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

LABELS = ["no", "intrinsic", "extrinsic"]

def evaluate_debate_results(pred_csv, test_jsonl):
    """Đánh giá kết quả debate"""
    print(f"📂 Loading predictions from {pred_csv}...")
    df_pred = pd.read_csv(pred_csv)
    
    print(f"📂 Loading ground truth from {test_jsonl}...")
    df_gt = pd.read_json(test_jsonl, lines=True)
    
    # Merge predictions với ground truth
    df_merged = df_pred.merge(df_gt[['id', 'label']], on='id', how='inner')
    
    print(f"📊 Evaluating {len(df_merged)} samples...")
    
    # Lấy predictions và ground truth
    preds = df_merged['predict_label'].str.lower().str.strip()
    gts = df_merged['label'].str.lower().str.strip()
    
    # Tính metrics
    acc = accuracy_score(gts, preds)
    f1m = f1_score(gts, preds, average="macro", labels=LABELS, zero_division=0)
    
    print("=" * 50)
    print("📊 DEBATE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1m:.4f}")
    print("\nClassification Report:")
    print(classification_report(gts, preds, labels=LABELS, digits=4, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(gts, preds, labels=LABELS))
    
    # Phân tích chi tiết
    print("\n" + "=" * 50)
    print("📈 DETAILED ANALYSIS")
    print("=" * 50)
    
    # Đếm số lượng mỗi class
    pred_counts = preds.value_counts()
    gt_counts = gts.value_counts()
    
    print("\nPrediction distribution:")
    for label in LABELS:
        count = pred_counts.get(label, 0)
        pct = count / len(preds) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print("\nGround truth distribution:")
    for label in LABELS:
        count = gt_counts.get(label, 0)
        pct = count / len(gts) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # So sánh với single-pass nếu có
    single_pass_file = "preds_test_singlepass.csv"
    try:
        df_single = pd.read_csv(single_pass_file)
        df_single_merged = df_single.merge(df_gt[['id', 'label']], on='id', how='inner')
        
        if len(df_single_merged) > 0:
            single_preds = df_single_merged['predict_label'].str.lower().str.strip()
            single_acc = accuracy_score(gts, single_preds)
            single_f1m = f1_score(gts, single_preds, average="macro", labels=LABELS, zero_division=0)
            
            print(f"\n📊 COMPARISON WITH SINGLE-PASS:")
            print(f"Single-pass Accuracy: {single_acc:.4f}")
            print(f"Single-pass Macro-F1: {single_f1m:.4f}")
            print(f"Debate Accuracy: {acc:.4f}")
            print(f"Debate Macro-F1: {f1m:.4f}")
            print(f"Improvement in Accuracy: {acc - single_acc:+.4f}")
            print(f"Improvement in Macro-F1: {f1m - single_f1m:+.4f}")
        else:
            print(f"\n⚠️  No matching samples found in single-pass results")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\n⚠️  Could not compare with single-pass: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate debate results")
    parser.add_argument("--pred_csv", type=str, required=True, help="Path to prediction CSV file")
    parser.add_argument("--test_jsonl", type=str, default="data/test.jsonl", help="Path to test JSONL file")
    
    args = parser.parse_args()
    evaluate_debate_results(args.pred_csv, args.test_jsonl)
