"""Fixed judge: the official LongMemEval answer-check prompt and label rule."""
from __future__ import annotations

from dataclasses import dataclass

from bench.llm import ChatOutcome, ProxiedChat
from bench.prompts import judge_label, judge_prompt, prompt_hashes


@dataclass
class JudgeConfig:
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.0
    max_tokens: int = 10

    def describe(self) -> dict:
        return {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "prompt_template_sha256": {k: v for k, v in prompt_hashes().items() if k.startswith("judge_")}, "label_rule": "'yes' in response.lower()"}


@dataclass
class Verdict:
    correct: bool
    raw: str
    outcome: ChatOutcome


class Judge:
    def __init__(self, chat: ProxiedChat, cfg: JudgeConfig) -> None:
        self.chat = chat
        self.cfg = cfg

    async def judge(self, *, question_type: str, question: str, answer: str, response: str, abstention: bool, session_id: str) -> Verdict:
        prompt = judge_prompt(question_type, question, answer, response, abstention=abstention)
        out = await self.chat.complete(model=self.cfg.model, messages=[{"role": "user", "content": prompt}], operation="judge", session_id=session_id, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens, n=1)
        return Verdict(correct=judge_label(out.text), raw=out.text.strip(), outcome=out)
