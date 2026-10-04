"""Shared 24-order question expansion for screened data preparation."""

import json
from itertools import permutations
from pathlib import Path

from .data import digest

LABELS = "ABCD"


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".jsonl.tmp")
    temp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    temp.replace(path)


def expand(example):
    """Map new positions to old positions, preserving the answer text exactly."""
    if len(example["choices"]) != 4 or example["answer"] not in range(4):
        raise ValueError("Expected four choices and an answer index in 0..3")
    variants = []
    for index, order in enumerate(permutations(range(4))):
        choices = [example["choices"][i] for i in order]
        answer = order.index(example["answer"])
        variants.append(dict(
            example, id=example["id"] + "::perm:{:02d}".format(index),
            parent_id=example["id"], permutation_index=index,
            choice_order=list(order), original_answer=example["answer"],
            original_content_hash=example["content_hash"],
            choices=choices, answer=answer,
            content_hash=digest([example["question"], choices, answer])))
    return variants
