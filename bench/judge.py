"""Fixed judge.  Styles:
  longmemeval_official  official evaluate_qa.py prompts per question type; label = 'yes' in reply.lower()   (Arm A)
  mem0_unified          Mem0's memory-benchmarks unified prompt with <judge_thinking>; their verdict parser  (Arm B)
"""
from __future__ import annotations

from dataclasses import dataclass

from bench.llm import ChatOutcome, ProxiedChat
from bench.prompts import JUDGE_STYLES, judge_label, judge_prompt, mem0_judge_label, mem0_judge_prompt, prompt_hashes


@dataclass
class JudgeConfig:
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.0
    max_tokens: int = 10
    style: str = "longmemeval_official"

    def __post_init__(self) -> None:
        if self.style not in JUDGE_STYLES:
            raise ValueError(f"unknown judge style {self.style!r}; choose from {JUDGE_STYLES}")

    def describe(self) -> dict:
        hashes = prompt_hashes()
        keys = [k for k in hashes if k.startswith("judge_")] if self.style == "longmemeval_official" else ["judge_mem0_unified"]
        return {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "style": self.style, "prompt_template_sha256": {k: hashes[k] for k in keys}, "label_rule": "'yes' in response.lower()" if self.style == "longmemeval_official" else "last standalone yes/no line after </judge_thinking>, else last yes/no token, else startswith yes"}


@dataclass
class Verdict:
    correct: bool
    raw: str
    outcome: ChatOutcome


class Judge:
    def __init__(self, chat: ProxiedChat, cfg: JudgeConfig) -> None:
        self.chat = chat
        self.cfg = cfg

    def build_prompt(self, *, question_type: str, question: str, answer: str, response: str, abstention: bool) -> str:
        if self.cfg.style == "mem0_unified":
            return mem0_judge_prompt(question, answer, response)
        return judge_prompt(question_type, question, answer, response, abstention=abstention)

    def label(self, raw: str) -> bool:
        return mem0_judge_label(raw) if self.cfg.style == "mem0_unified" else judge_label(raw)

    async def judge(self, *, question_type: str, question: str, answer: str, response: str, abstention: bool, session_id: str) -> Verdict:
        prompt = self.build_prompt(question_type=question_type, question=question, answer=answer, response=response, abstention=abstention)
        out = await self.chat.complete(model=self.cfg.model, messages=[{"role": "user", "content": prompt}], operation="judge", session_id=session_id, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens, n=1)
        return Verdict(correct=self.label(out.text), raw=out.text.strip(), outcome=out)
