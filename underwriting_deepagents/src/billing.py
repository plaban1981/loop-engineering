BUDGET_USD = 2.00
INPUT_PRICE_PER_TOKEN = 3.00 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 15.00 / 1_000_000


class BudgetExhaustedError(Exception):
    pass


class CostMeter:
    def __init__(self, budget_usd: float = BUDGET_USD):
        self.spent = 0.0
        self._budget = budget_usd

    def record(self, input_tokens: int, output_tokens: int) -> None:
        if self.spent >= self._budget:
            raise BudgetExhaustedError(f"Budget ${self._budget:.2f} exhausted at ${self.spent:.4f}")
        self.spent += input_tokens * INPUT_PRICE_PER_TOKEN + output_tokens * OUTPUT_PRICE_PER_TOKEN

    def check(self) -> None:
        """Raise BudgetExhaustedError if the budget is already exhausted. Call before each agent.invoke()."""
        if self.spent >= self._budget:
            raise BudgetExhaustedError(f"Budget ${self._budget:.2f} exhausted at ${self.spent:.4f}")

    def record_from_message(self, msg) -> None:
        if getattr(msg, "usage_metadata", None):
            self.record(
                msg.usage_metadata.get("input_tokens", 0),
                msg.usage_metadata.get("output_tokens", 0),
            )

    def print_receipt(self, train_score=None, test_score=None, seed_test_score=None):
        print(f"\nspend: ${self.spent:.4f}")
        if train_score is not None:
            print(f"train score: {train_score:.0%}")
        if test_score is not None and seed_test_score is not None:
            print(f"test score: {test_score:.0%}  gain: {test_score - seed_test_score:+.0%}")
