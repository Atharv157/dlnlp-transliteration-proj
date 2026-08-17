#!/usr/bin/env python3
import json
import argparse

##########################################################
# 🔹 Evaluation Metric Functions (from NEWS)
##########################################################

def LCS_length(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def f_score(candidate, references):
    if not candidate or not references:
        return 0.0, ''
    best_ref = references[0]
    best_ref_lcs = LCS_length(candidate, best_ref)
    for ref in references[1:]:
        lcs = LCS_length(candidate, ref)
        if (len(ref) - 2 * lcs) < (len(best_ref) - 2 * best_ref_lcs):
            best_ref, best_ref_lcs = ref, lcs
    precision = best_ref_lcs / len(candidate) if candidate else 0.0
    recall = best_ref_lcs / len(best_ref) if best_ref else 0.0
    if precision + recall == 0:
        return 0.0, best_ref
    return 2 * precision * recall / (precision + recall), best_ref


def mean_average_precision(candidates, references, n):
    total, num_correct = 0.0, 0
    for k in range(n):
        if k < len(candidates) and candidates[k] in references:
            num_correct += 1
        total += num_correct / (k + 1)
    return total / n


def inverse_rank(candidates, reference):
    for i, cand in enumerate(candidates):
        if cand == reference:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(predictions, references):
    acc, f, f_best, mrr, map_ref = {}, {}, {}, {}, {}
    for src_word, refs in references.items():
        preds = predictions.get(src_word, [])
        if preds:
            acc[src_word] = 1.0 if preds[0] in refs else 0.0
            f[src_word], f_best[src_word] = f_score(preds[0], refs)
            mrr[src_word] = max(inverse_rank(preds, ref) for ref in refs)
            map_ref[src_word] = mean_average_precision(preds, refs, len(refs))
        else:
            acc[src_word], f[src_word], f_best[src_word] = 0.0, 0.0, ''
            mrr[src_word], map_ref[src_word] = 0.0, 0.0
    return acc, f, f_best, mrr, map_ref

##########################################################
# 🔹 Data Loading
##########################################################

def load_jsonl_data(file_path, key_roman="english word", key_native="native word"):
    """Load data from JSONL file into dict[roman_word] = [native_word]"""
    print(f"📖 Loading data from: {file_path}")
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                roman = obj.get(key_roman, "").strip()
                dev = obj.get(key_native, "").strip()
                if roman and dev:
                    data[roman.upper()] = [dev.upper()]
            except json.JSONDecodeError:
                continue
    print(f"✅ Loaded {len(data)} entries from {file_path}")
    return data

##########################################################
# 🔹 Main Entry
##########################################################

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM transliteration outputs against gold references")
    parser.add_argument("--test-file", type=str, required=True, help="Path to gold reference JSONL file")
    parser.add_argument("--pred-file", type=str, required=True, help="Path to LLM predictions JSONL file")
    args = parser.parse_args()

    # Load gold and predicted data
    gold_data = load_jsonl_data(args.test_file)
    pred_data = load_jsonl_data(args.pred_file)

    # Align predictions to gold (only evaluate where test words exist)
    predictions = {}
    for roman_word in gold_data.keys():
        if roman_word in pred_data:
            predictions[roman_word] = pred_data[roman_word]
        else:
            predictions[roman_word] = [""]

    # Evaluate metrics
    acc, f, f_best, mrr, map_ref = evaluate(predictions, gold_data)
    N = len(acc)
    acc_score = sum(acc.values()) / N
    f_score_val = sum(f.values()) / N
    mrr_score = sum(mrr.values()) / N
    map_ref_score = sum(map_ref.values()) / N

    print("\n📊 FINAL EVALUATION RESULTS (LLM OUTPUTS)")
    print("=" * 50)
    print(f"Word-level Exact Accuracy (ACC):          {acc_score:.4f}")
    print(f"Character-level F1 Score (Mean F-score):  {f_score_val:.4f}")
    print(f"MRR:                                      {mrr_score:.4f}")
    print(f"MAP_ref:                                  {map_ref_score:.4f}")
    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    main()
