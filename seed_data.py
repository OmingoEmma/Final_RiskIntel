import random
from datetime import datetime, timedelta

from db import init_db, insert_interaction, insert_feedback


def seed_interactions(n: int = 200) -> None:
    init_db()
    now = datetime.utcnow()
    for i in range(n):
        ts = now - timedelta(minutes=random.randint(0, 72 * 60))
        feature = "summarize"
        model = random.choice(["RuleBased v1", "Echo v1"]) 
        text_len = random.randint(15, 200)
        user_text = " ".join(["word"] * text_len)
        output_text = " ".join(["summary"] * max(5, text_len // random.randint(3, 6)))
        latency_ms = max(30, int(random.gauss(180, 60)))
        success = random.random() > 0.05
        input_tokens = len(user_text.split())
        output_tokens = len(output_text.split())
        insert_interaction(
            session_id=f"seed-{random.randint(1, 20)}",
            timestamp=ts,
            feature=feature,
            user_input=user_text,
            model_output=output_text,
            model_name=model,
            latency_ms=latency_ms,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def seed_feedback(n: int = 60) -> None:
    init_db()
    now = datetime.utcnow()
    for i in range(n):
        ts = now - timedelta(minutes=random.randint(0, 72 * 60))
        feature = "summarize"
        usability = min(5, max(1, int(random.gauss(4.2, 0.7))))
        accuracy = min(5, max(1, int(random.gauss(4.0, 0.8))))
        recommend = random.random() < 0.8
        comments = random.choice([
            "Works well for short inputs.",
            "Would like more control over summary length.",
            "Sometimes repeats phrases.",
            "Fast and easy to use!",
            "Accuracy could be better on technical text.",
            None,
        ])
        insert_feedback(
            timestamp=ts,
            session_id=f"seed-{random.randint(1, 20)}",
            feature=feature,
            usability_rating=usability,
            accuracy_rating=accuracy,
            would_recommend=recommend,
            comments=comments,
        )


if __name__ == "__main__":
    print("Seeding interactions and feedback...")
    seed_interactions()
    seed_feedback()
    print("Done. Data written to data/app.db")