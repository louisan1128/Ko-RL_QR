
FAILURE_TYPE_POLICY = {
    "expression_mismatch": ["keyword", "expanded"],
    "ellipsis": ["structured", "expanded"],
    "compound_noun": ["keyword"],
    "colloquial_mismatch": ["prompt"],
    "abbreviation": ["expanded"],
    "unlabeled": ["original", "keyword", "expanded"],
}


def select_strategies(failure_type: str) -> list[str]:
    return FAILURE_TYPE_POLICY.get(failure_type, ["original", "keyword", "expanded"])
