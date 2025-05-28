import argparse

def clean_context(context):
    """清理 context，去掉多餘的空白，將英文轉成小寫，返回去重後的詞列表"""
    return list(set([word.strip().lower() if word.isascii() else word.strip() for word in context.split(",") if word.strip()]))

def calculate_precision_recall_f1(ref_context_path, hyp_context_path, output_missed_path):
    with open(ref_context_path, "r", encoding="utf-8") as ref_file:
        ref_lines = [line.strip().split("\t") for line in ref_file.readlines()]
    with open(hyp_context_path, "r", encoding="utf-8") as hyp_file:
        hyp_lines = [line.strip().split("\t") for line in hyp_file.readlines()]

    assert len(ref_lines) == len(hyp_lines), "Mismatch in number of sentences between ref and hyp files."

    precision_scores = []
    recall_scores = []
    f1_scores = []

    missed_entities = []  # 用於記錄每個句子的漏掉詞

    for ref, hyp in zip(ref_lines, hyp_lines):
        ref_id, ref_context = ref[0], clean_context(ref[1]) if len(ref) > 1 else []
        hyp_id, hyp_context = hyp[0], clean_context(hyp[1]) if len(hyp) > 1 else []

        assert ref_id == hyp_id, f"ID mismatch: {ref_id} != {hyp_id}"

        # 計算 Precision, Recall, F1
        ref_set = set(ref_context)
        hyp_set = set(hyp_context)

        tp = len(ref_set & hyp_set)  # True Positives
        fp = len(hyp_set - ref_set)  # False Positives
        fn = len(ref_set - hyp_set)  # False Negatives

        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # 記錄漏掉的詞
        missed_words = ref_set - hyp_set
        if missed_words:
            missed_entities.append(f"{ref_id}\t{', '.join(missed_words)}")

    # 計算 Macro-Averaged Precision, Recall, F1
    macro_precision = sum(precision_scores) / len(precision_scores)
    macro_recall = sum(recall_scores) / len(recall_scores)
    macro_f1 = sum(f1_scores) / len(f1_scores)

    print(f"Precision (Macro-Averaged): {macro_precision:.2f}%")
    print(f"Recall (Macro-Averaged): {macro_recall:.2f}%")
    print(f"F1 Score (Macro-Averaged): {macro_f1:.2f}%")

    # 將漏掉的詞寫入檔案
    if missed_entities:
        with open(output_missed_path, "w", encoding="utf-8") as output_file:
            output_file.write("\n".join(missed_entities))
        print(f"Missed entities saved to: {output_missed_path}")
    else:
        print("No missed entities.")

    return macro_precision, macro_recall, macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Precision, Recall, and F1 Score for context retrieval and save missed entities.")
    parser.add_argument('--ref_context_path', type=str, required=True, help='Path to reference context file.')
    parser.add_argument('--hyp_context_path', type=str, required=True, help='Path to hypothesis context file.')
    parser.add_argument('--output_missed_path', type=str, required=True, help='Path to output missed entities file.')
    args = parser.parse_args()

    calculate_precision_recall_f1(args.ref_context_path, args.hyp_context_path, args.output_missed_path)
