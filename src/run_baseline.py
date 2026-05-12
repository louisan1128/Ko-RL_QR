import argparse
from pathlib import Path

import pandas as pd

from src.evaluate import evaluate_retriever, load_qa_pairs, save_hard_cases
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval baseline evaluation for Ko-RL-QR.")
    parser.add_argument("--corpus", required=True, help="Path to processed corpus.jsonl.")
    parser.add_argument("--qa", required=True, help="Path to processed qa_pairs.jsonl.")
    parser.add_argument("--retriever", required=True, choices=["bm25", "dense", "hybrid"], help="Retriever type to evaluate.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to retrieve for evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of QA pairs for quick experiments.")
    parser.add_argument("--alpha", type=float, nargs="*", default=None, help="Alpha value(s) for hybrid retriever.")
    parser.add_argument("--embedding_dir", default="embeddings", help="Directory to store dense embeddings and index.")
    parser.add_argument("--output_csv", default="results/original_baseline.csv", help="CSV output path for baseline metrics.")
    parser.add_argument("--hard_cases_out", default="results/hard_cases.jsonl", help="Output path for hard cases JSONL.")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.output_csv).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    qa_pairs = load_qa_pairs(args.qa)
    if args.limit:
        qa_pairs = qa_pairs[: args.limit]

    rows = []
    hard_cases_written = False

    if args.retriever == "bm25":
        retriever = BM25Retriever(args.corpus)
        metrics = evaluate_retriever(retriever, qa_pairs, top_k=args.top_k)
        rows.append(
            {
                "retriever": "bm25",
                "alpha": "N/A",
                "recall@1": metrics["recall@1"],
                "recall@5": metrics["recall@5"],
                "recall@10": metrics["recall@10"],
                "mrr": metrics["mrr"],
                "num_questions": metrics["num_questions"],
            }
        )
        save_hard_cases(retriever, qa_pairs, args.hard_cases_out, top_k=args.top_k)
        hard_cases_written = True

    elif args.retriever == "dense":
        retriever = DenseRetriever(args.corpus, embedding_dir=args.embedding_dir)
        metrics = evaluate_retriever(retriever, qa_pairs, top_k=args.top_k)
        rows.append(
            {
                "retriever": "dense",
                "alpha": "N/A",
                "recall@1": metrics["recall@1"],
                "recall@5": metrics["recall@5"],
                "recall@10": metrics["recall@10"],
                "mrr": metrics["mrr"],
                "num_questions": metrics["num_questions"],
            }
        )
        save_hard_cases(retriever, qa_pairs, args.hard_cases_out, top_k=args.top_k)
        hard_cases_written = True

    elif args.retriever == "hybrid":
        alpha_values = args.alpha if args.alpha else [0.3, 0.5, 0.7]
        bm25_retriever = BM25Retriever(args.corpus)
        dense_retriever = DenseRetriever(args.corpus, embedding_dir=args.embedding_dir)

        for alpha in alpha_values:
            retriever = HybridRetriever(bm25_retriever, dense_retriever, alpha=alpha)
            metrics = evaluate_retriever(retriever, qa_pairs, top_k=args.top_k)
            rows.append(
                {
                    "retriever": "hybrid",
                    "alpha": alpha,
                    "recall@1": metrics["recall@1"],
                    "recall@5": metrics["recall@5"],
                    "recall@10": metrics["recall@10"],
                    "mrr": metrics["mrr"],
                    "num_questions": metrics["num_questions"],
                }
            )
            file_suffix = f"_alpha_{alpha:.2f}" if len(alpha_values) > 1 else ""
            hard_cases_file = Path(args.hard_cases_out)
            hard_cases_path = hard_cases_file.with_name(hard_cases_file.stem + file_suffix + hard_cases_file.suffix)
            save_hard_cases(retriever, qa_pairs, hard_cases_path, top_k=args.top_k)
            if alpha == 0.5 or len(alpha_values) == 1:
                save_hard_cases(retriever, qa_pairs, args.hard_cases_out, top_k=args.top_k)
                hard_cases_written = True

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved baseline metrics to {args.output_csv}.")
    if hard_cases_written:
        print(f"Saved hard cases to {args.hard_cases_out}.")


if __name__ == "__main__":
    main()
