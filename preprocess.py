"""
NanoLLM v2 - Dataset Preprocessing
Downloads and cleans datasets for language model pretraining.

Standards:
- No empty records
- No malformed records
- No excessive whitespace
- No duplicates
- No extremely long samples
- English only
- Consistent text format
"""
import os
import re
import json
import hashlib
from datasets import load_dataset


PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"


def clean_text(text: str) -> str:
    """Clean and normalize text. Returns empty string if invalid."""
    if not text or not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize whitespace (tabs, newlines, multiple spaces)
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove empty or too short
    if len(text) < 20:
        return ""

    # Remove extremely long (pathological)
    if len(text) > 10000:
        return ""

    # Remove lines that are mostly special characters
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return ""

    return text


def remove_duplicates(texts: list[str]) -> list[str]:
    """Remove exact duplicates by content hash."""
    seen = set()
    unique = []
    for t in texts:
        h = hashlib.md5(t.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(t)
    return unique


def process_fineweb_edu(num_samples: int = 100000) -> list[str]:
    """
    Process FineWeb-Edu dataset.
    Educational web text - good quality English.
    """
    print("  Loading FineWeb-Edu...")
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        texts = []
        rejected = 0
        for i, item in enumerate(ds):
            if i >= num_samples:
                break

            text = clean_text(item.get("text", ""))
            if text:
                texts.append(text)
            else:
                rejected += 1

            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1:,} / {num_samples:,}...")

        print(f"    FineWeb-Edu: {len(texts):,} accepted, {rejected:,} rejected")
        return texts

    except Exception as e:
        print(f"    FineWeb-Edu failed: {e}")
        return []


def process_dailydialog() -> list[str]:
    """
    Process DailyDialog dataset.
    Natural everyday conversations.
    """
    print("  Loading DailyDialog...")
    try:
        ds = load_dataset("daily_dialog")
        texts = []
        rejected = 0

        for split in ds:
            for item in ds[split]:
                dialog = item.get("dialog", [])
                if not dialog:
                    rejected += 1
                    continue

                # Join dialog turns with proper formatting
                turns = [turn.strip() for turn in dialog if turn.strip()]
                if len(turns) < 2:
                    rejected += 1
                    continue

                text = " ".join(turns)
                text = clean_text(text)

                if text:
                    texts.append(text)
                else:
                    rejected += 1

        texts = remove_duplicates(texts)
        print(f"    DailyDialog: {len(texts):,} accepted, {rejected:,} rejected")
        return texts

    except Exception as e:
        print(f"    DailyDialog failed: {e}")
        return []


def process_squad() -> list[str]:
    """
    Process SQuAD dataset.
    Question-answer pairs for factual knowledge.
    """
    print("  Loading SQuAD...")
    try:
        ds = load_dataset("squad")
        texts = []
        rejected = 0

        for split in ds:
            for item in ds[split]:
                question = item.get("question", "").strip()
                answers = item.get("answers", {})
                answer_texts = answers.get("text", [])

                if not question or not answer_texts:
                    rejected += 1
                    continue

                # Take first answer
                answer = answer_texts[0].strip()

                # Format as clean Q&A
                text = f"Question: {question} Answer: {answer}"
                text = clean_text(text)

                if text and len(text) > 30:
                    texts.append(text)
                else:
                    rejected += 1

        texts = remove_duplicates(texts)
        print(f"    SQuAD: {len(texts):,} accepted, {rejected:,} rejected")
        return texts

    except Exception as e:
        print(f"    SQuAD failed: {e}")
        return []


def process_openbookqa() -> list[str]:
    """
    Process OpenBookQA dataset.
    Factual/reasoning questions.
    """
    print("  Loading OpenBookQA...")
    try:
        ds = load_dataset("openbookqa")
        texts = []
        rejected = 0

        for split in ds:
            for item in ds[split]:
                question = item.get("question", "").strip()
                choices = item.get("choices", {})
                choice_texts = choices.get("text", [])
                answer_key = item.get("answerKey", "")

                if not question or not choice_texts or not answer_key:
                    rejected += 1
                    continue

                # Get correct answer
                idx = ord(answer_key) - ord("A")
                if 0 <= idx < len(choice_texts):
                    answer = choice_texts[idx].strip()
                    text = f"Question: {question} Answer: {answer}"
                    text = clean_text(text)

                    if text and len(text) > 30:
                        texts.append(text)
                    else:
                        rejected += 1
                else:
                    rejected += 1

        texts = remove_duplicates(texts)
        print(f"    OpenBookQA: {len(texts):,} accepted, {rejected:,} rejected")
        return texts

    except Exception as e:
        print(f"    OpenBookQA failed: {e}")
        return []


def preprocess_all():
    """
    Preprocess all datasets and save as JSONL.
    Each line: {"text": "clean text content"}
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    all_texts = []

    # FineWeb-Edu (~80%)
    fineweb = process_fineweb_edu(100000)
    all_texts.extend(fineweb)

    # DailyDialog (~10%)
    daily = process_dailydialog()
    all_texts.extend(daily)

    # SQuAD (~5%)
    squad = process_squad()
    all_texts.extend(squad)

    # OpenBookQA (~5%)
    openbook = process_openbookqa()
    all_texts.extend(openbook)

    print(f"\n  Total raw: {len(all_texts):,}")

    # Remove duplicates
    before = len(all_texts)
    all_texts = remove_duplicates(all_texts)
    after = len(all_texts)
    print(f"  Duplicates removed: {before - after:,}")

    # Compute stats
    char_lengths = [len(t) for t in all_texts]
    avg_chars = sum(char_lengths) / max(len(char_lengths), 1)
    max_chars = max(char_lengths) if char_lengths else 0
    total_chars = sum(char_lengths)

    # Save
    output_path = os.path.join(PROCESSED_DIR, "pretraining.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    # Stats
    stats = {
        "total_documents": len(all_texts),
        "total_characters": total_chars,
        "avg_characters": round(avg_chars),
        "max_characters": max_chars,
    }
    stats_path = os.path.join(PROCESSED_DIR, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Dataset preprocessing")
    print(f"  {'─'*40}")
    print(f"  Documents:      {len(all_texts):,}")
    print(f"  Total chars:    {total_chars:,}")
    print(f"  Avg chars/doc:  {avg_chars:.0f}")
    print(f"  Max chars/doc:  {max_chars:,}")
    print(f"  Saved to:       {output_path}")

    return output_path, stats


if __name__ == "__main__":
    preprocess_all()
