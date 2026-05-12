# Ko-RL-QR

Ko-RL-QR은 한국어 RAG 환경에서 원본 질문이 검색기에 최적화되어 있지 않을 수 있는 문제를 다루는 프로젝트입니다. 1주차 목표는 original question baseline retrieval을 구축하고 BM25, Dense, Hybrid retriever의 성능을 측정하는 것입니다.

## Week 1 목표

- KorQuAD 또는 KLUE-MRC 스타일 QA 데이터를 retrieval 실험용으로 전처리
- `corpus.jsonl`과 `qa_pairs.jsonl` 생성
- BM25 retriever 구현
- Dense retriever 구현
- Hybrid retriever 구현
- Recall@1, Recall@5, Recall@10, MRR 평가 코드 구현
- original query baseline 결과 CSV로 저장
- top-10 안에 gold document를 찾지 못한 hard cases 저장

## 폴더 구조

```
ko-rl-qr/
  data/
    raw/
    processed/
      corpus.jsonl
      qa_pairs.jsonl
  src/
    __init__.py
    preprocess.py
    evaluate.py
    run_baseline.py
    retrievers/
      __init__.py
      bm25.py
      dense.py
      hybrid.py
  embeddings/
  results/
    original_baseline.csv
    hard_cases.jsonl
  requirements.txt
  README.md
```

## 데이터 파일 형식

### corpus.jsonl

각 줄은 JSON object이며 필수 키는 `doc_id`, `text`입니다.

예시:

```json
{"doc_id": "doc_000001", "text": "세종대왕은 1397년에 태어났으며 조선의 제4대 왕이다."}
```

### qa_pairs.jsonl

각 줄은 JSON object이며 필수 키는 `qid`, `question`, `answer`, `gold_doc_id`입니다.

예시:

```json
{"qid": "q_000001", "question": "세종대왕은 언제 태어났는가?", "answer": "1397년", "gold_doc_id": "doc_000001"}
```

`gold_doc_id`는 반드시 `corpus.jsonl`의 `doc_id`와 일치해야 합니다.

## 실행 방법

### 데이터 전처리

```bash
python src/preprocess.py --raw_path data/raw/KorQuAD.json --corpus_out data/processed/corpus.jsonl --qa_out data/processed/qa_pairs.jsonl --limit 1000
```

### baseline 실행

```bash
python src/run_baseline.py --retriever bm25 --corpus data/processed/corpus.jsonl --qa data/processed/qa_pairs.jsonl --top_k 10

python src/run_baseline.py --retriever dense --corpus data/processed/corpus.jsonl --qa data/processed/qa_pairs.jsonl --top_k 10

python src/run_baseline.py --retriever hybrid --alpha 0.5 --corpus data/processed/corpus.jsonl --qa data/processed/qa_pairs.jsonl --top_k 10
```

## 결과 파일 설명

- `results/original_baseline.csv`: retriever별 원본 질문 baseline 결과. 열은 `retriever`, `alpha`, `recall@1`, `recall@5`, `recall@10`, `mrr`, `num_questions`입니다.
- `results/hard_cases.jsonl`: original question으로 top-10 안에 gold document를 찾지 못한 문제를 저장한 파일입니다.
