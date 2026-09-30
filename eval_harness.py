
import argparse
import sys
import requests

eval_prompts = {
    "cancel_order": "I need to cancel an order I placed yesterday",
    "change_order": "can I still edit my order before it ships",
    "change_shipping_address": "i moved, need to update where my stuff gets delivered",
    "check_cancellation_fee": "is there a fee if I cancel now",
    "check_invoice": "where can I see my invoice for last month",
    "check_payment_methods": "what ways can I pay you guys",
    "check_refund_policy": "what's your policy on refunds",
    "complaint": "this is the third time my order came late, im frustrated",
    "contact_customer_service": "how do I reach your support team",
    "contact_human_agent": "can I talk to an actual person not a bot",
    "create_account": "how do i sign up for an account",
    "delete_account": "please close my account",
    "delivery_options": "what delivery choices do you have",
    "delivery_period": "how many days will delivery take",
    "edit_account": "i need to update my account info",
    "get_invoice": "send me a copy of my invoice",
    "get_refund": "i want my money back",
    "newsletter_subscription": "sign me up for your emails",
    "payment_issue": "my card got charged twice",
    "place_order": "how do i buy something from you",
    "recover_password": "cant remember my password",
    "registration_problems": "signup form keeps giving an error",
    "review": "want to leave a review for a product i bought",
    "set_up_shipping_address": "need to add a new delivery address",
    "switch_account": "i want to change from basic to premium",
    "track_order": "wheres my package",
    "track_refund": "checking on the status of my refund",
    "OUT_OF_DOMAIN": "can you recommend a good recipe for pasta",
    "MIXED_INTENT": "cancel one item in my order and also update my shipping address for the rest",
    "ADVERSARIAL_TYPOS": "y cnat i cancell da ordr???",
}

THRESHOLD = 0.70


def run_eval(endpoint: str, smoke: bool = False) -> tuple[int, int]:
    # smoke = every 3rd prompt (1/3 of the full list), evenly spread rather than
    # just the first few, so it still covers a mix of intents and edge cases
    items = list(eval_prompts.items())[::3] if smoke else list(eval_prompts.items())
    valid_intents = {k for k, _ in items if not k.isupper()}

    results = []
    for expected_label, prompt in items:
        try:
            resp = requests.post(endpoint, json={"user_query": prompt}, timeout=600)
            data = resp.json()
        except Exception as e:
            print(f"REQUEST FAILED for {expected_label!r}: {e}")
            results.append((expected_label, "REQUEST_FAILED"))
            continue

        outcome = data.get("guardrail_outcome")
        intent = data.get("intent")

        if expected_label in valid_intents:
            status = "CORRECT" if (outcome == "pass" and intent == expected_label) else "WRONG"
        else:
            status = "CORRECTLY_HANDLED" if outcome == "fallback" else "SLIPPED_THROUGH"

        results.append((expected_label, status))
        print(f"{expected_label:28s} -> {status:20s} (outcome={outcome}, intent={intent})")

    n_correct = sum(1 for label, status in results if label in valid_intents and status == "CORRECT")
    print(f"\n{n_correct}/{len(valid_intents)} real intents correctly classified")
    return n_correct, len(valid_intents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--smoke", action="store_true", help="run 1/3 subset instead of full 30")
    args = parser.parse_args()

    n_correct, n_total = run_eval(args.endpoint, smoke=args.smoke)
    pass_rate = n_correct / n_total
    print(f"Pass rate: {pass_rate:.1%} (threshold: {THRESHOLD:.0%})")

    if pass_rate < THRESHOLD:
        print("FAILED")
        sys.exit(1)
    print("PASSED")
    sys.exit(0)