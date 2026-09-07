"""Fixed reader model: answers the question from the memory context supplied by the adapter."""
from __future__ import annotations

from dataclasses import dataclass

from bench.llm import ChatOutcome, ProxiedChat
from bench.prompts import READER_TEMPLATES, reader_prompt, sha


@dataclass
class ReaderConfig:
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.0
    max_tokens: int = 500
    prompt: str = "longmemeval_facts"

    def describe(self) -> dict:
        return {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "prompt": self.prompt, "prompt_template_sha256": sha(READER_TEMPLATES[self.prompt])}


class Reader:
    def __init__(self, chat: ProxiedChat, cfg: ReaderConfig) -> None:
        self.chat = chat
        self.cfg = cfg

    def build_prompt(self, context: str, question_date: str, question: str) -> str:
        return reader_prompt(self.cfg.prompt, context, question_date, question)

    async def answer(self, *, context: str, question_date: str, question: str, session_id: str) -> ChatOutcome:
        prompt = self.build_prompt(context, question_date, question)
        return await self.chat.complete(model=self.cfg.model, messages=[{"role": "user", "content": prompt}], operation="answer", session_id=session_id, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
