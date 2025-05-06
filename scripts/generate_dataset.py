"""Generate a realistic, reproducible support-ticket dataset.

This bundled dataset lets the whole pipeline run offline (great for CI and for
anyone cloning the repo). Tickets are composed from interchangeable fragments
plus filler and light noise, so the task requires genuine generalisation rather
than memorising templates -- the baseline lands in a believable F1 range.

For real training, point `data.py` at a public dataset (see README, e.g.
Banking77); the rest of the pipeline is unchanged.

Usage:
    python scripts/generate_dataset.py --n 1200 --out data/tickets.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

OPENERS = [
    "Hi team,", "Hello,", "Hey,", "Good morning,", "To whom it may concern,",
    "Quick question -", "I need some help.", "", "Urgent:", "Following up:",
]
CLOSERS = [
    "Thanks in advance.", "Please advise.", "Appreciate any help.", "",
    "Let me know.", "This is blocking me.", "Hoping for a quick reply.",
    "Regards.", "Sorry if this is the wrong channel.",
]

CORES: dict[str, list[str]] = {
    "billing": [
        "I was charged {amount} twice this month and need the duplicate refunded",
        "my subscription jumped to {amount} without any notice",
        "I can't find the invoice for {month} and my finance team needs it",
        "the {amount} charge on my statement doesn't look right",
        "I was billed {amount} even though my card was declined",
        "the promo code didn't apply so I paid the full {amount}",
        "I'd like a refund on the annual plan I bought in {month}",
        "please confirm whether I'll be charged {amount} again next cycle",
        "there are two separate {amount} charges from the same day",
    ],
    "technical": [
        "the app crashes whenever I upload a file over 10MB",
        "I keep getting a 500 error when I hit export on the dashboard",
        "sync has been broken since the update in {month}",
        "pages take over 30 seconds to load lately",
        "the mobile app freezes on the login screen",
        "API calls time out intermittently and break my integration",
        "the analytics charts render completely blank in Safari",
        "the CSV download comes out corrupted and won't open",
        "notifications stopped firing after the latest release",
    ],
    "account_access": [
        "I can't log in and the password reset email never arrives",
        "my account got locked after a few failed attempts",
        "I need to change the email address on my account",
        "two-factor isn't sending me a code anymore",
        "I see logins I don't recognise and I'm worried about my account",
        "I want to add a teammate with admin permissions",
        "I was removed from my workspace by mistake",
        "the forgot-password page says my account doesn't exist",
        "I'm stuck in a redirect loop trying to sign in with SSO",
    ],
    "feature_request": [
        "it would be great to have a dark mode",
        "could you support exporting to PDF and not just CSV",
        "please add a way to bulk-edit records",
        "an integration with Slack would be really useful",
        "do you have plans to support SSO for enterprise",
        "a mobile widget for quick stats would help a lot",
        "I'd love to schedule reports to send automatically each week",
        "let us customise which columns show in the main table",
        "an API webhook for new events would unblock our team",
    ],
    "general_inquiry": [
        "what are your support hours and how do I reach a person",
        "do you offer discounts for non-profits",
        "where can I find docs for the reporting feature",
        "is there a free trial before I commit",
        "how does your product compare to others I'm evaluating",
        "can you point me to a getting-started tutorial",
        "what's your data retention and privacy policy",
        "do you publish a public roadmap I can follow",
        "which plan is right for a team of about ten people",
    ],
}

AMOUNTS = ["$9.99", "$19", "$29", "$49.00", "$120", "$199", "$15.50", "$80"]
MONTHS = ["January", "March", "April", "June", "September", "last month", "Q1"]


def _typo(text: str, rng: random.Random) -> str:
    if rng.random() > 0.15 or len(text) < 12:
        return text
    i = rng.randint(1, len(text) - 2)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


def compose(label: str, rng: random.Random) -> str:
    core = rng.choice(CORES[label]).format(
        amount=rng.choice(AMOUNTS), month=rng.choice(MONTHS)
    )
    parts = [rng.choice(OPENERS), core.capitalize() + ".", rng.choice(CLOSERS)]
    text = " ".join(p for p in parts if p).strip()
    return _typo(text, rng)


def generate(n: int, seed: int, noise: float) -> list[tuple[str, str]]:
    """Build rows. `noise` is the fraction of labels flipped to a different
    class to mimic real annotator disagreement on ambiguous tickets, which keeps
    the task non-trivial (and makes the quality/monitoring story meaningful)."""
    rng = random.Random(seed)
    labels = list(CORES)
    rows = []
    for _ in range(n):
        true_label = rng.choice(labels)
        text = compose(true_label, rng)
        if rng.random() < noise:
            assigned = rng.choice([x for x in labels if x != true_label])
        else:
            assigned = true_label
        rows.append((text, assigned))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1200, help="number of rows")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--noise", type=float, default=0.10,
                        help="fraction of labels flipped (annotator noise)")
    parser.add_argument("--out", type=Path, default=Path("data/tickets.csv"))
    args = parser.parse_args()

    rows = generate(args.n, args.seed, args.noise)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
