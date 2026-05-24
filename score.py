import argparse
import json
import sys
from pathlib import Path

from vector_store import VectorStore


def parse_chunk_id(chunk_id: str):
    parts = chunk_id.rsplit(":", 2)
    if len(parts) != 3:
        return None
    path, name, line_str = parts
    try:
        lineno = int(line_str)
    except ValueError:
        return None
    return path, name, lineno


def chunks_match(predicted: str, reference: str, tolerance: int = 2) -> bool:
    p = parse_chunk_id(predicted)
    r = parse_chunk_id(reference)
    if p is None or r is None:
        return False
    p_path, p_name, p_line = p
    r_path, r_name, r_line = r
    return p_path == r_path and p_name == r_name and abs(p_line - r_line) <= tolerance


def score_question(top5: list[str], correct: list[str]) -> float:
    if len(correct) == 0:
        return 0.0

    seen = []
    for chunk in top5:
        if chunk not in seen:
            seen.append(chunk)
    top5_dedup = seen

    matched = 0
    used_refs = set()
    for pred in top5_dedup:
        for i, ref in enumerate(correct):
            if i not in used_refs and chunks_match(pred, ref):
                matched += 1
                used_refs.add(i)
                break

    return matched / min(5, len(correct))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VectorStore against questions"
    )
    parser.add_argument(
        "--questions", required=True, help="Path to eval_questions.json"
    )
    parser.add_argument(
        "--predictions", help="Output or input results.json", default="results.json"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Do not run inference, just evaluate existing predictions",
    )
    args = parser.parse_args()

    q_path = Path(args.questions)
    if not q_path.exists():
        print(f"Error: {q_path} not found")
        sys.exit(1)

    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not args.evaluate_only:
        print("Running inference using VectorStore...")
        store = VectorStore()
        predictions = []
        for q in questions:
            qid = q["question_id"]
            query = q["query"]
            results = store.search(query, top_k=5)
            predictions.append(
                {"question_id": qid, "top_5_chunks": [r.chunk_id for r in results]}
            )

        with open(args.predictions, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"Saved predictions to {args.predictions}")

    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    gt = {q["question_id"]: q for q in questions}
    pred_index = {p["question_id"]: p["top_5_chunks"] for p in predictions}

    per_question = []
    for qid, q in gt.items():
        correct = q.get("correct_chunk_ids", [])
        top5 = pred_index.get(qid, [])
        score = score_question(top5, correct)
        per_question.append(
            {
                "question_id": qid,
                "difficulty": q.get("difficulty", "unknown"),
                "language": q.get("language", "unknown"),
                "n_correct": len(correct),
                "score": score,
            }
        )

    total = len(per_question)
    if total == 0:
        print("No questions evaluated.")
        return

    mean_score = sum(r["score"] for r in per_question) / total

    print("=== CodeLens RAG -- Scoring ===")
    print(f"Questions evaluated: {total}")
    print(f"Mean Precision@5: {mean_score:.3f}")

    print("\nPer-question detail:")
    for r in sorted(per_question, key=lambda x: x["question_id"]):
        n = r["n_correct"]
        s = r["score"]
        matched_count = round(s * min(5, r["n_correct"]))
        print(
            f"  {r['question_id']} [{r['difficulty']}, {r['language']}]"
            f" -- {matched_count}/{min(5, n)} -> {s:.2f}"
        )


if __name__ == "__main__":
    main()
