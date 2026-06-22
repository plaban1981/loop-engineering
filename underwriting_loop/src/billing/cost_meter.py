# src/billing/cost_meter.py

BUDGET_USD = 2.00
INPUT_PRICE_PER_TOKEN = 3.00 / 1_000_000    # claude-sonnet-4-6 input pricing
OUTPUT_PRICE_PER_TOKEN = 15.00 / 1_000_000  # claude-sonnet-4-6 output pricing


class BudgetExhaustedError(Exception):
    pass


class CostMeter:
    def __init__(self, budget_usd: float = BUDGET_USD):
        self.spent = 0.0
        self._budget = budget_usd

    def record(self, input_tokens: int, output_tokens: int) -> None:
        if self.spent >= self._budget:
            raise BudgetExhaustedError(
                f"Budget ${self._budget:.2f} exhausted at ${self.spent:.4f}"
            )
        cost = input_tokens * INPUT_PRICE_PER_TOKEN + output_tokens * OUTPUT_PRICE_PER_TOKEN
        self.spent += cost

    def record_from_message(self, msg) -> None:
        if getattr(msg, "usage_metadata", None):
            self.record(
                input_tokens=msg.usage_metadata.get("input_tokens", 0),
                output_tokens=msg.usage_metadata.get("output_tokens", 0),
            )

    def print_receipt(
        self,
        train_score: float = None,
        test_score: float = None,
        seed_test_score: float = None,
    ) -> None:
        print(f"\nspend:              ${self.spent:.4f}")
        if train_score is not None:
            print(f"train score:        {train_score:.0%}")
        if test_score is not None and seed_test_score is not None:
            gain = test_score - seed_test_score
            print(f"test score:         {test_score:.0%}")
            print(f"gain (vs baseline): {gain:+.0%}")
            if self.spent > 0 and train_score:
                print(f"cost per point:     ${self.spent / max(train_score, 0.01):.4f}")
