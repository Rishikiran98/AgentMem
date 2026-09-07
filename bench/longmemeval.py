"""LongMemEval dataset model, loading, seeded selection, and a synthetic stand-in for tests.

Official data: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
(``longmemeval_s_cleaned.json``; the original ``longmemeval_s.json`` shares the schema).
Schema (README "Dataset Format"): question_id, question_type, question, answer,
question_date, haystack_session_ids, haystack_dates, haystack_sessions
(list of sessions; each a list of {role, content[, has_answer]}), answer_session_ids.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
)


@dataclass
class Turn:
    role: str
    content: str
    has_answer: bool = False


@dataclass
class Session:
    session_id: str
    date: str
    index: int
    turns: list[Turn]


@dataclass
class Instance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: list[Session]
    answer_session_ids: list[str] = field(default_factory=list)

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def n_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)


@dataclass
class Dataset:
    name: str
    path: str
    sha256: str
    size_bytes: int
    instances: list[Instance]
    synthetic: bool = False

    def by_id(self) -> dict[str, Instance]:
        return {i.question_id: i for i in self.instances}


class DatasetError(ValueError):
    pass


def _parse_instance(raw: dict[str, Any], idx: int) -> Instance:
    required = ("question_id", "question_type", "question", "answer", "question_date", "haystack_session_ids", "haystack_dates", "haystack_sessions")
    missing = [k for k in required if k not in raw]
    if missing:
        raise DatasetError(f"instance {idx} missing fields {missing}")
    if raw["question_type"] not in QUESTION_TYPES:
        raise DatasetError(f"instance {raw['question_id']}: unknown question_type {raw['question_type']!r}")
    ids, dates, sessions = raw["haystack_session_ids"], raw["haystack_dates"], raw["haystack_sessions"]
    if not (len(ids) == len(dates) == len(sessions)):
        raise DatasetError(f"instance {raw['question_id']}: haystack lists differ in length")
    sess: list[Session] = []
    for j, (sid, date, turns) in enumerate(zip(ids, dates, sessions)):
        parsed = []
        for t in turns:
            if not isinstance(t, dict) or "role" not in t or "content" not in t:
                raise DatasetError(f"instance {raw['question_id']} session {sid}: malformed turn")
            parsed.append(Turn(role=str(t["role"]), content=str(t["content"]), has_answer=bool(t.get("has_answer", False))))
        sess.append(Session(session_id=str(sid), date=str(date), index=j, turns=parsed))
    return Instance(
        question_id=str(raw["question_id"]),
        question_type=str(raw["question_type"]),
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        question_date=str(raw["question_date"]),
        sessions=sess,
        answer_session_ids=[str(x) for x in raw.get("answer_session_ids", [])],
    )


def load_longmemeval(path: str | Path, *, name: str | None = None) -> Dataset:
    p = Path(path)
    raw_bytes = p.read_bytes()
    try:
        data = json.loads(raw_bytes)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw_bytes.decode().splitlines() if line.strip()]
    if not isinstance(data, list) or not data:
        raise DatasetError(f"{p}: expected a non-empty JSON list of instances")
    instances = [_parse_instance(x, i) for i, x in enumerate(data)]
    ids = [i.question_id for i in instances]
    if len(set(ids)) != len(ids):
        raise DatasetError(f"{p}: duplicate question_id values")
    synthetic = bool(isinstance(data[0], dict) and data[0].get("_synthetic"))
    return Dataset(name=name or p.stem, path=str(p), sha256=hashlib.sha256(raw_bytes).hexdigest(), size_bytes=len(raw_bytes), instances=instances, synthetic=synthetic)


def select_instances(dataset: Dataset, *, seed: int, limit: int | None = None, question_ids: Iterable[str] | None = None) -> list[Instance]:
    """Seeded question ordering.  ``limit`` takes the first N of the seeded order, so a
    subset is reproducible from (dataset sha256, seed, limit) alone."""
    by_id = dataset.by_id()
    if question_ids:
        wanted = list(question_ids)
        unknown = [q for q in wanted if q not in by_id]
        if unknown:
            raise DatasetError(f"unknown question ids: {unknown}")
        pool = [by_id[q] for q in wanted]
    else:
        pool = list(dataset.instances)
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    ordered = [pool[i] for i in order]
    return ordered[:limit] if limit else ordered


def selection_digest(instances: list[Instance]) -> str:
    return hashlib.sha256("\n".join(i.question_id for i in instances).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------- #
# Synthetic stand-in (tests and offline demos only; never for paper numbers)
# ---------------------------------------------------------------------- #

_ATTRS = [("favorite color", "teal"), ("cat's name", "Pixel"), ("hometown", "Coimbra"), ("car", "a green Volvo"), ("sister's name", "Ines"), ("favorite dish", "lentil stew"), ("running shoe brand", "Saucony"), ("piano teacher", "Mr Alves")]
_FILLER = ["Can you explain how tides work?", "What is a good recipe for bread?", "How do I set up a python virtualenv?", "Recommend a book about astronomy.", "How far is the moon?", "What's the capital of Peru?"]


def make_synthetic_dataset(path: str | Path, *, n_instances: int = 4, seed: int = 0, sessions_per_instance: int = 3, turns_per_session: int = 4) -> Dataset:
    """Write a small LongMemEval-schema file with one needle fact per instance."""
    rng = random.Random(seed)
    out = []
    for i in range(n_instances):
        attr, value = _ATTRS[i % len(_ATTRS)]
        qtype = QUESTION_TYPES[i % len(QUESTION_TYPES)]
        abstention = i % 7 == 6
        needle_session = rng.randrange(sessions_per_instance)
        sessions, dates, sids = [], [], []
        for s in range(sessions_per_instance):
            turns = []
            for t in range(turns_per_session // 2):
                q = rng.choice(_FILLER)
                turns.append({"role": "user", "content": q})
                turns.append({"role": "assistant", "content": f"Sure. Here is an explanation about: {q.lower().rstrip('?')}."})
            if s == needle_session and not abstention:
                turns.insert(1, {"role": "user", "content": f"By the way, my {attr} is {value}.", "has_answer": True})
                turns.insert(2, {"role": "assistant", "content": f"Nice, I will remember that your {attr} is {value}."})
            sessions.append(turns)
            dates.append(f"2023/05/{10 + s:02d} (Wed) 10:{i:02d}")
            sids.append(f"syn_{i}_s{s}")
        qid = f"syn_{i}" + ("_abs" if abstention else "")
        out.append(
            {
                "_synthetic": True,
                "question_id": qid,
                "question_type": qtype,
                "question": f"What is my {attr}?",
                "answer": "The user never mentioned this." if abstention else value,
                "question_date": f"2023/06/01 (Thu) 09:0{i % 10}",
                "haystack_session_ids": sids,
                "haystack_dates": dates,
                "haystack_sessions": sessions,
                "answer_session_ids": [] if abstention else [sids[needle_session]],
            }
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return load_longmemeval(p, name="synthetic-longmemeval")
