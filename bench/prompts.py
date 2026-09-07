"""Fixed reader and judge prompts, copied verbatim from the official LongMemEval code.

Source: https://github.com/xiaowu0162/LongMemEval (branch main, fetched 2026-09-07)
  src/evaluation/evaluate_qa.py   sha256 ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251
  src/generation/run_generation.py sha256 4f1eb3c69d7ad40f04065b9c0bc86f6582441018fc6ff751d162d66c95baf672

Judge settings in the official script: model gpt-4o-2024-08-06, a single user
message, n=1, temperature=0, max_tokens=10, label = 'yes' in response.lower().
Reader settings: single user message, temperature=0, max_tokens=500 (non-CoT).
"""
from __future__ import annotations

import hashlib

# --- reader (run_generation.py, prepare_prompt) -------------------------- #
# Variant used when the retrieval output is *facts* rather than raw sessions
# (merge_key_expansion_into_value == 'replace', cot=False).  Memory systems
# return extracted memories, so this is the matching official template.
READER_FACTS = "I will give you several facts extracted from history chats between you and a user. Please answer the question based on the relevant facts.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"
# Variant used when raw sessions are provided (merge_key_expansion_into_value None, cot=False).
READER_SESSIONS = "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"

READER_TEMPLATES = {"longmemeval_facts": READER_FACTS, "longmemeval_sessions": READER_SESSIONS}

# --- judge (evaluate_qa.py, get_anscheck_prompt) -------------------------- #
JUDGE_DEFAULT = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
JUDGE_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
JUDGE_KNOWLEDGE_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
JUDGE_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
JUDGE_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."


def judge_prompt(task: str, question: str, answer: str, response: str, *, abstention: bool) -> str:
    """Exact port of ``get_anscheck_prompt``."""
    if abstention:
        return JUDGE_ABSTENTION.format(question, answer, response)
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return JUDGE_DEFAULT.format(question, answer, response)
    if task == "temporal-reasoning":
        return JUDGE_TEMPORAL.format(question, answer, response)
    if task == "knowledge-update":
        return JUDGE_KNOWLEDGE_UPDATE.format(question, answer, response)
    if task == "single-session-preference":
        return JUDGE_PREFERENCE.format(question, answer, response)
    raise NotImplementedError(task)


def judge_label(response_text: str) -> bool:
    """Exact port of the official label rule."""
    return "yes" in response_text.strip().lower()


def reader_prompt(template_name: str, context: str, question_date: str, question: str) -> str:
    return READER_TEMPLATES[template_name].format(context, question_date, question)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_hashes() -> dict[str, str]:
    return {
        "reader_longmemeval_facts": sha(READER_FACTS),
        "reader_longmemeval_sessions": sha(READER_SESSIONS),
        "judge_default": sha(JUDGE_DEFAULT),
        "judge_temporal": sha(JUDGE_TEMPORAL),
        "judge_knowledge_update": sha(JUDGE_KNOWLEDGE_UPDATE),
        "judge_preference": sha(JUDGE_PREFERENCE),
        "judge_abstention": sha(JUDGE_ABSTENTION),
    }
