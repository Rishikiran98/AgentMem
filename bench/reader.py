"""Fixed reader model: answers the question from the memory context supplied by the adapter.

Two prompt families are supported, selected by ``ReaderConfig.prompt``:
  longmemeval_facts / longmemeval_sessions  official LongMemEval templates (Arm A)
  mem0_v3                                   Mem0's published answerer prompt + post-processing (Arm B)
"""
from __future__ import annotations

from dataclasses import dataclass

from adapters.base import ReadHit
from bench.llm import ChatOutcome, ProxiedChat
from bench.prompts import READER_TEMPLATES, mem0_format_memories, mem0_human_date, mem0_reader_prompt, mem0_strip_answer, reader_prompt, sha


@dataclass
class ReaderConfig:
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.0
    max_tokens: int = 500
    prompt: str = "longmemeval_facts"
    memory_order: str = "system"  # system | created_at_asc  (Arm B sorts the cutoff slice by created_at)

    def describe(self) -> dict:
        return {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "prompt": self.prompt, "memory_order": self.memory_order, "prompt_template_sha256": sha(READER_TEMPLATES[self.prompt])}


@dataclass
class ReaderResult:
    outcome: ChatOutcome
    answer: str      # what the judge sees
    answer_raw: str  # verbatim model output (differs from answer only for mem0_v3 post-processing)
    prompt: str


class Reader:
    def __init__(self, chat: ProxiedChat, cfg: ReaderConfig) -> None:
        self.chat = chat
        self.cfg = cfg

    def build_prompt(self, *, context: str, hits: list[ReadHit], question_date: str, question: str) -> str:
        if self.cfg.prompt == "mem0_v3":
            ordered = list(hits)
            if self.cfg.memory_order == "created_at_asc":
                ordered = sorted(ordered, key=lambda h: h.created_at or "")
            block = mem0_format_memories([(h.text, h.created_at) for h in ordered])
            return mem0_reader_prompt(block, mem0_human_date(question_date), question)
        return reader_prompt(self.cfg.prompt, context, question_date, question)

    async def answer(self, *, context: str, question_date: str, question: str, session_id: str, hits: list[ReadHit] | None = None) -> ReaderResult:
        prompt = self.build_prompt(context=context, hits=hits or [], question_date=question_date, question=question)
        out = await self.chat.complete(model=self.cfg.model, messages=[{"role": "user", "content": prompt}], operation="answer", session_id=session_id, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
        answer = mem0_strip_answer(out.text) if self.cfg.prompt == "mem0_v3" else out.text
        return ReaderResult(outcome=out, answer=answer, answer_raw=out.text, prompt=prompt)
