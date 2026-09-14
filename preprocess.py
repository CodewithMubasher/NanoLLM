"""
NanoLLM v2 - Dataset Preprocessing
Downloads and preprocesses datasets for language model pretraining.

Datasets:
- FineWeb-Edu (~80%): Educational web text
- DailyDialog (~10%): Natural conversations
- SQuAD (~5%): Question-answer pairs
- OpenBookQA (~5%): Factual questions
"""
import os
import json
import re
import hashlib
from datasets import load_dataset


PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 10:
        return ""
    if len(text) > 50000:
        return ""
    return text


def remove_duplicates(texts: list[str]) -> list[str]:
    """Remove exact duplicates by hash."""
    seen = set()
    unique = []
    for t in texts:
        h = hashlib.md5(t.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(t)
    return unique


def process_fineweb_edu(num_samples: int = 200000) -> list[str]:
    """Process FineWeb-Edu dataset."""
    print("Loading FineWeb-Edu...")
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        texts = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            text = clean_text(item.get("text", ""))
            if text:
                texts.append(text)
        print(f"  FineWeb-Edu: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  FineWeb-Edu failed: {e}")
        return []


def process_dailydialog() -> list[str]:
    """Process DailyDialog dataset."""
    print("Loading DailyDialog...")
    try:
        ds = load_dataset("daily_dialog")
        texts = []
        for split in ds:
            for item in ds[split]:
                dialog = item.get("dialog", [])
                if dialog:
                    text = " ".join(dialog)
                    text = clean_text(text)
                    if text:
                        texts.append(text)
        texts = remove_duplicates(texts)
        print(f"  DailyDialog: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  DailyDialog failed: {e}")
        return []


def process_squad() -> list[str]:
    """Process SQuAD dataset."""
    print("Loading SQuAD...")
    try:
        ds = load_dataset("squad")
        texts = []
        for split in ds:
            for item in ds[split]:
                q = item.get("question", "")
                a = " ".join(item.get("answers", {}).get("text", []))
                text = clean_text(f"{q} {a}")
                if text:
                    texts.append(text)
        texts = remove_duplicates(texts)
        print(f"  SQuAD: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  SQuAD failed: {e}")
        return []


def process_openbookqa() -> list[str]:
    """Process OpenBookQA dataset."""
    print("Loading OpenBookQA...")
    try:
        ds = load_dataset("openbookqa")
        texts = []
        for split in ds:
            for item in ds[split]:
                q = item.get("question", "")
                choices = item.get("choices", {}).get("text", [])
                answer = item.get("answerKey", "")
                if choices and answer:
                    idx = ord(answer) - ord("A")
                    if 0 <= idx < len(choices):
                        text = clean_text(f"{q} {choices[idx]}")
                        if text:
                            texts.append(text)
        texts = remove_duplicates(texts)
        print(f"  OpenBookQA: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  OpenBookQA failed: {e}")
        return []


def preprocess_all():
    """Preprocess all datasets and save as JSONL."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_texts = []

    fineweb = process_fineweb_edu(200000)
    all_texts.extend(fineweb)

    daily = process_dailydialog()
    all_texts.extend(daily)

    squad = process_squad()
    all_texts.extend(squad)

    openbook = process_openbookqa()
    all_texts.extend(openbook)

    print(f"\nTotal raw samples: {len(all_texts)}")

    before = len(all_texts)
    all_texts = remove_duplicates(all_texts)
    after = len(all_texts)

    print(f"Duplicates removed: {before - after}")

    stats = {
        "raw": before,
        "accepted": after,
        "duplicates_removed": before - after,
        "avg_chars": sum(len(t) for t in all_texts) / max(len(all_texts), 1),
        "max_chars": max(len(t) for t in all_texts) if all_texts else 0,
    }

    output_path = os.path.join(PROCESSED_DIR, "pretraining.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    stats_path = os.path.join(PROCESSED_DIR, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset preprocessing")
    print(f"{'='*40}")
    print(f"Raw examples:      {stats['raw']:,}")
    print(f"Accepted:          {stats['accepted']:,}")
    print(f"Duplicates removed:{stats['duplicates_removed']:,}")
    print(f"Average characters:{stats['avg_chars']:.0f}")
    print(f"Maximum characters:{stats['max_chars']:,}")
    print(f"\nSaved to {output_path}")

    return output_path, stats


if __name__ == "__main__":
    preprocess_all()
