"""
Prepare teacher-forced train/dev JSON files from answer-annotated QA data.

The script:
1. Loads a full answer-annotated dataset.
2. Excludes question_ids from a final test input file.
3. Splits the remaining examples into train/dev at the question/video level.
4. Expands each answer into a user -> assistant conversation sample that
   generate_training_data.py can consume.
"""

import argparse
import json
import os
import random
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare teacher-forced train/dev splits")
    parser.add_argument("--full_data", type=str, required=True,
                        help="Answer-annotated source JSON, e.g. ego-proactivevideoqa_format.json")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Final test JSON whose question_ids must be excluded")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_by", type=str, default="question_id",
                        choices=["question_id", "video", "none"],
                        help="Key used to exclude final test examples. Use none for exploratory split only.")
    parser.add_argument("--split_by", type=str, default="question_id",
                        choices=["question_id", "video"],
                        help="Group key for train/dev split to avoid leakage")
    parser.add_argument("--include_system", action="store_true",
                        help="Preserve a system turn if present in the source conversation")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_id(example: Dict, fallback_prefix: str) -> str:
    return str(example.get("question_id") or example.get("video") or fallback_prefix)


def get_group_key(example: Dict, split_by: str, fallback: str) -> str:
    value = example.get(split_by)
    if value is not None:
        return str(value)
    return get_id(example, fallback)


def first_user_turn(conversation: List[Dict]):
    for turn in conversation:
        if turn.get("role") == "user":
            return turn
    return None


def expand_answers(examples: List[Dict], include_system: bool) -> List[Dict]:
    expanded = []
    skipped_no_user = 0
    skipped_no_answer = 0

    for ex_idx, ex in enumerate(examples):
        conversation = ex.get("conversation") or ex.get("messages") or []
        user_turn = first_user_turn(conversation)
        if user_turn is None:
            skipped_no_user += 1
            continue

        answers = ex.get("answer") or ex.get("answers") or []
        if isinstance(answers, dict):
            answers = [answers]
        if not answers:
            skipped_no_answer += 1
            continue

        system_turns = []
        if include_system:
            system_turns = [turn for turn in conversation if turn.get("role") == "system"][:1]

        base_id = get_id(ex, f"sample_{ex_idx}")
        for ans_idx, ans in enumerate(answers):
            if isinstance(ans, str):
                content = ans.strip()
            elif isinstance(ans, dict):
                content = str(ans.get("content", "")).strip()
            else:
                continue
            if not content:
                continue

            sample = {
                "question_id": f"{base_id}_ans{ans_idx}",
                "video": ex.get("video"),
                "duration": ex.get("duration"),
                "conversation": [
                    *system_turns,
                    {
                        "role": "user",
                        "time": user_turn.get("time", 0),
                        "content": user_turn.get("content", ""),
                    },
                    {
                        "role": "assistant",
                        "content": content,
                    },
                ],
            }
            expanded.append(sample)

    return expanded, skipped_no_user, skipped_no_answer


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    full_data = load_json(args.full_data)
    if not isinstance(full_data, list):
        raise ValueError("--full_data must be a JSON list")

    test_ids = set()
    if args.test_data and args.exclude_by != "none":
        test_data = load_json(args.test_data)
        if not isinstance(test_data, list):
            raise ValueError("--test_data must be a JSON list")
        if args.exclude_by == "question_id":
            test_ids = {get_id(ex, f"test_{idx}") for idx, ex in enumerate(test_data)}
        else:
            test_ids = {
                str(ex.get("video"))
                for ex in test_data
                if ex.get("video") is not None
            }

    filtered = [
        ex for idx, ex in enumerate(full_data)
        if (
            args.exclude_by == "none"
            or (
                get_id(ex, f"sample_{idx}") if args.exclude_by == "question_id"
                else str(ex.get("video"))
            ) not in test_ids
        )
    ]
    if not filtered:
        raise ValueError(
            "No examples remain after excluding test_data. This usually means full_data "
            "and test_data refer to the same set. For an exploratory train/dev split, "
            "rerun with --exclude_by none and omit final-test claims."
        )

    groups = {}
    for idx, ex in enumerate(filtered):
        key = get_group_key(ex, args.split_by, f"sample_{idx}")
        groups.setdefault(key, []).append(ex)

    group_keys = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(group_keys)

    n_dev = max(1, int(len(group_keys) * args.dev_ratio)) if group_keys else 0
    dev_keys = set(group_keys[:n_dev])

    train_examples = []
    dev_examples = []
    for key, items in groups.items():
        if key in dev_keys:
            dev_examples.extend(items)
        else:
            train_examples.extend(items)

    train_expanded, train_no_user, train_no_answer = expand_answers(train_examples, args.include_system)
    dev_expanded, dev_no_user, dev_no_answer = expand_answers(dev_examples, args.include_system)

    train_path = os.path.join(args.outdir, "teacher_forced_train.json")
    dev_path = os.path.join(args.outdir, "teacher_forced_dev.json")
    meta_path = os.path.join(args.outdir, "teacher_forced_split_meta.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_expanded, f, ensure_ascii=False, indent=2)
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_expanded, f, ensure_ascii=False, indent=2)

    meta = {
        "full_data": args.full_data,
        "test_data": args.test_data,
        "exclude_by": args.exclude_by,
        "split_by": args.split_by,
        "dev_ratio": args.dev_ratio,
        "seed": args.seed,
        "full_examples": len(full_data),
        "excluded_test_ids": len(test_ids),
        "remaining_examples": len(filtered),
        "groups": len(group_keys),
        "train_groups": len(group_keys) - len(dev_keys),
        "dev_groups": len(dev_keys),
        "train_examples_before_expand": len(train_examples),
        "dev_examples_before_expand": len(dev_examples),
        "train_samples": len(train_expanded),
        "dev_samples": len(dev_expanded),
        "train_skipped_no_user": train_no_user,
        "train_skipped_no_answer": train_no_answer,
        "dev_skipped_no_user": dev_no_user,
        "dev_skipped_no_answer": dev_no_answer,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"train: {train_path}")
    print(f"dev:   {dev_path}")
    if train_expanded:
        print("first train sample:")
        print(json.dumps(train_expanded[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
