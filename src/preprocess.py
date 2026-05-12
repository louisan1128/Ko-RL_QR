import argparse
import json
from pathlib import Path


def load_korquad(raw_path):
    """Load KorQuAD-style JSON data from raw_path."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    with raw_path.open("r", encoding="utf-8") as reader:
        data = json.load(reader)

    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data

    raise ValueError(
        f"Unsupported KorQuAD format: expected top-level dict with 'data' or list, got {type(data).__name__}"
    )


def _write_jsonl(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_corpus_and_qa(raw_path, corpus_out, qa_out, limit=None):
    """Build corpus.jsonl and qa_pairs.jsonl from KorQuAD-style raw data."""
    data_items = load_korquad(raw_path)

    corpus_records = []
    qa_records = []
    context_to_doc_id = {}
    doc_counter = 1
    q_counter = 1

    for article in data_items:
        paragraphs = article.get("paragraphs") or article.get("paragraph", [])
        if not isinstance(paragraphs, list):
            raise ValueError("Each article must contain a 'paragraphs' list.")

        for paragraph in paragraphs:
            context = paragraph.get("context") or paragraph.get("text")
            if not isinstance(context, str) or not context.strip():
                raise ValueError("Each paragraph must contain a non-empty 'context' string.")

            if context not in context_to_doc_id:
                doc_id = f"doc_{doc_counter:06d}"
                context_to_doc_id[context] = doc_id
                corpus_records.append({"doc_id": doc_id, "text": context})
                doc_counter += 1
            else:
                doc_id = context_to_doc_id[context]

            qas = paragraph.get("qas") or paragraph.get("questions") or []
            if not isinstance(qas, list):
                raise ValueError("Paragraph 'qas' field must be a list.")

            for qa in qas:
                question = qa.get("question") or qa.get("query")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("Each QA item must contain a non-empty 'question'.")

                qid = qa.get("id") or f"q_{q_counter:06d}"
                answers = qa.get("answers") or qa.get("answer") or []
                answer_text = ""

                if isinstance(answers, list) and answers:
                    first_answer = answers[0]
                    if isinstance(first_answer, dict):
                        answer_text = first_answer.get("text", "")
                    else:
                        answer_text = str(first_answer)
                elif isinstance(answers, str):
                    answer_text = answers

                qa_records.append(
                    {
                        "qid": qid,
                        "question": question,
                        "answer": answer_text,
                        "gold_doc_id": doc_id,
                    }
                )
                q_counter += 1

                if limit and len(qa_records) >= limit:
                    break

            if limit and len(qa_records) >= limit:
                break

        if limit and len(qa_records) >= limit:
            break

    if not corpus_records:
        raise ValueError("No corpus records were created from the raw data.")
    if not qa_records:
        raise ValueError("No QA pairs were created from the raw data.")

    _write_jsonl(corpus_records, corpus_out)
    _write_jsonl(qa_records, qa_out)

    return corpus_records, qa_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess KorQuAD-style JSON into corpus and QA JSONL files.")
    parser.add_argument("--raw_path", required=True, help="Input raw KorQuAD JSON path.")
    parser.add_argument("--corpus_out", required=True, help="Output corpus.jsonl path.")
    parser.add_argument("--qa_out", required=True, help="Output qa_pairs.jsonl path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of QA pairs for quick experiments.")
    args = parser.parse_args()

    build_corpus_and_qa(args.raw_path, args.corpus_out, args.qa_out, limit=args.limit)
    print(f"Saved corpus to {args.corpus_out} and QA pairs to {args.qa_out}.")
