from dataclasses import dataclass

from src.utils.text import tokenize


@dataclass
class RewardCalculator:
    alpha: float = 1.0
    beta: float = 0.5
    lambda_: float = 0.05

    def compute_reward(self, recall_score: float, mrr_score: float, query: str) -> float:
        # TODO: Add answer F1, faithfulness, or downstream generation quality terms.
        length_penalty = self.lambda_ * len(tokenize(query))
        return self.alpha * recall_score + self.beta * mrr_score - length_penalty
