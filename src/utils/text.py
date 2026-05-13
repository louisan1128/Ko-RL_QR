import re


KOREAN_STOPWORDS = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "에서",
    "에게",
    "으로",
    "로",
    "와",
    "과",
    "의",
    "도",
    "만",
    "그리고",
    "또는",
    "무엇",
    "누구",
    "언제",
    "어디",
    "어떤",
    "몇",
    "왜",
}

SYNONYM_EXPANSIONS = {
    "세종대왕": ["세종", "조선", "훈민정음"],
    "훈민정음": ["한글", "창제", "세종"],
    "한글": ["훈민정음", "문자", "창제"],
    "수도": ["도읍", "중심지", "행정"],
    "독립": ["해방", "자주", "운동"],
    "저자": ["작가", "쓴 사람"],
}


def tokenize(text: str) -> list[str]:
    """Tokenize Korean text with a dependency-light rule-based tokenizer."""

    if not text:
        return []
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_]+", " ", text)
    raw_tokens = [token for token in cleaned.strip().split() if token]
    return [_normalize_token(token) for token in raw_tokens if _normalize_token(token)]


def _normalize_token(token: str) -> str:
    token = token.lower().strip()
    for suffix in ("으로", "에서", "에게", "까지", "부터", "이다", "였다", "했다", "하는", "하며", "이고"):
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            return token[: -len(suffix)]
    for suffix in ("은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "로"):
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def extract_keywords(text: str, max_keywords: int = 6) -> str:
    # TODO: Replace this with a Korean morphological analyzer such as Kiwi, Mecab, or Okt.
    tokens = tokenize(text)
    keywords = [token for token in tokens if len(token) > 1 and token not in KOREAN_STOPWORDS]
    deduped = list(dict.fromkeys(keywords))
    return " ".join(deduped[:max_keywords])


def prompt_style_query(question: str) -> str:
    question = question.strip()
    return f"{question}에 대한 정답 근거 문서를 찾아줘"


def expanded_query(question: str) -> str:
    # TODO: Replace this dictionary with LLM, thesaurus, or corpus-mined expansion terms.
    tokens = tokenize(question)
    expansions = []
    for token in tokens:
        expansions.extend(SYNONYM_EXPANSIONS.get(token, []))
    keywords = extract_keywords(question)
    expanded_terms = " ".join(dict.fromkeys(expansions))
    return " ".join(part for part in [question.strip(), keywords, expanded_terms] if part)


def structured_query(question: str) -> str:
    keywords = extract_keywords(question)
    return f"entity: {keywords} relation: 관련 정보 target: 정답 근거 문서"
