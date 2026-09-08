# Milestone 4: Mem0 LongMemEval-S reproduction (pre-registration)

Status: **pre-registered, not yet executed.** This document was written and
committed before any run against the real dataset or a real provider. It fixes
the reference numbers, the comparison rule, the two experimental arms, and the
discrepancies already identifiable from source, so that the eventual result is
interpreted by rules chosen in advance.

## 1. The published result

| Source | Figure | Setting |
|---|---|---|
| Mem0 README (`github.com/mem0ai/mem0`, `main`, fetched 2026-09-07), "New Memory Algorithm (April 2026)" | **LongMemEval 94.4** (old algorithm 67.8); 6.8K tokens; latency p50 1.09 s | "Scores reflect Mem0's managed platform, which includes proprietary optimizations not available in the open-source SDK; open-source users should expect directionally similar gains but not identical numbers." Single-pass retrieval at a top_200 budget. |
| `github.com/mem0ai/memory-benchmarks` README, "Mem0 Platform" | **94.4% (472/500)** at top-200; 94.8% (474/500) at top-50; per type: knowledge-update 93.6, multi-session 88.0, single-session-assistant 98.2, single-session-preference 96.7, single-session-user 98.6, temporal-reasoning 97.0 | Platform, v3 pipeline. |
| same README, "OSS with Different Extraction Models" | GPT-5 extraction **91.0%**; GPT-OSS-120B 89.8; Llama 4 Maverick 88.6; Gemma 4 31B 88.6 | Self-hosted OSS pipeline, Qwen 600M embedder via SageMaker, GPT-5 answerer and judge. No result is published for the default OpenAI stack (gpt-4o-mini extraction, text-embedding-3-small). |
| `docs/core-concepts/memory-evaluation.mdx` (mem0 repo) | same 94.4 and per-type table; "Mean tokens: 6,787" | "Scores carry a ±1 point confidence interval due to judge inconsistency." |

The type counts (78+133+56+30+70+133 = 500) match LongMemEval-S, and their
runner downloads `longmemeval_s_cleaned.json`, so the figure is on the 2025-09
cleaned LongMemEval-S file.

**There is no published Mem0 number under the official LongMemEval protocol
(official reader prompt, official judge prompts).** Every published figure uses
Mem0's own answerer and judge prompts.

## 2. Two pre-registered arms

| | Arm A: official protocol | Arm B: Mem0's published protocol |
|---|---|---|
| system config | `configs/mem0.yaml` | `configs/mem0-published-protocol.yaml` (top_k 200) |
| benchmark config | `configs/benchmarks/longmemeval_s.yaml` | `configs/benchmarks/longmemeval_s_mem0protocol.yaml` |
| ingestion | one `add()` per haystack session, session date as leading system message | one `add()` per user/assistant pair (their `pair_turns`), no date conveyed (their OSS server forwards none), empty pairs skipped |
| retrieval | `search(question, top_k=20, threshold=0.1)` (mem0ai defaults) | `search(question, top_k=200, threshold=0.1)` |
| reader | official LongMemEval facts template, gpt-4o-2024-08-06, T=0, 500 tokens | Mem0 `ANSWER_GENERATION_PROMPT` verbatim, memories grouped by `created_at` date, `<mem_thinking>` stripped and `ANSWER:` split, gpt-4o, T=0, 4096 tokens |
| judge | official `evaluate_qa.py` prompts per type + abstention, gpt-4o-2024-08-06, T=0, 10 tokens, label = "yes" in reply | Mem0 unified `JUDGE_PROMPT` verbatim, gpt-4o, T=0, 4096 tokens, their `_parse_yes_no_judgment` |
| question of interest | What does Mem0 OSS score under the benchmark's own rules? (new measurement) | Does Mem0 OSS 2.0.20 with the default OpenAI stack reproduce 94.4 under Mem0's rules? (reproduction) |

Both arms run through the same adapter, proxy, settlement check, and event
logging. Prompt hashes for both families are in every `RUN_START`.

Model choice for Arm B: the memory-benchmarks README documents gpt-4o as the
answerer/judge default; the runner's argparse default is gpt-5 and the OSS
table used GPT-5. We pre-register **gpt-4o** (documented default, and the same
family as Arm A so the two arms differ in prompts and depth, not reader
strength). If gpt-4o fails to reproduce, a gpt-5 Arm B' may be run as a
*follow-up*, labelled as such; it does not replace Arm B.

## 3. Comparison rule (fixed in advance)

Let `p` be the run's accuracy over judged instances with a Wilson 95%
interval `[lo, hi]`, and `r = 0.944` the published value.

* **reproduced** if `|p − r| ≤ 0.03` **or** `lo ≤ r ≤ hi`.
* **not_reproduced** otherwise.
* Instances that errored before judging are reported separately; the headline
  uses judged instances and a second figure counts errors as wrong.
* Seeds: Arm B once at seed 42 over all 500 instances (Mem0's own run is a
  single pass); Arm A at seeds 42, 43, 44 for the confidence interval used
  in the paper. Question order is seeded; it does not affect accuracy for a
  sequential run but keeps the traces reproducible.
* No configuration is changed after a verdict is seen. A configuration bug
  found afterwards is a new commit that invalidates the affected run ids.

The 3-point tolerance is deliberately wider than the vendor's ±1: it absorbs
judge-model non-determinism and the OSS/platform gap the vendor itself
acknowledges. If Arm B lands outside even this, the discrepancy is real.

## 4. Discrepancies identifiable before running

These come from reading the published code; each is a row in the automatic
checklist emitted by `scripts/reproduce_mem0.py`.

1. **Platform vs OSS.** The 94.4 is from the managed platform. The vendor
   states OSS "should expect directionally similar gains but not identical
   numbers." A strict reproduction with OSS is therefore not expected to
   succeed by the vendor's own account.
2. **Unreproducible software pin.** Their OSS server installs
   `mem0ai @ git+https://github.com/mem0ai/mem0.git@feat/v3-pipeline`. That
   branch returns 404 today; the exact commit is unrecorded. We use the PyPI
   release `mem0ai==2.0.20` (2026-09-02), which contains the v3 pipeline.
3. **Their OSS server is incompatible with the released library.** It calls
   `search(query, limit=…, user_id=…)`; mem0ai 2.0.20 rejects top-level
   `user_id` and ignores `limit` (the parameter is `top_k`, default 20). It
   also never forwards `timestamp`/`observation_date` to `add()`, so under
   OSS the session dates never reach extraction and `created_at` is the
   ingestion time, while the answer prompt groups memories by `created_at`
   as if it were the conversation date. Arm B mirrors the OSS behaviour
   (`date_mode: none`) and sets `top_k=200` explicitly.
4. **Prompts.** Mem0's answer prompt contains rules that name specific
   LongMemEval items ("chandelier counts as jewelry", "scratch grains count
   as new layer feed", "potlucks count as dinner parties", "starting a
   diorama project counts as working on that model kit"). Its judge prompt
   instructs the judge that it "has a tendency to say no too quickly" and to
   "lean toward yes". Neither is the official protocol; both raise measured
   accuracy relative to the official prompts by construction. Arm A quantifies
   the gap.
5. **Judge output length.** Official: 10 tokens, "yes" substring. Mem0: free
   chain-of-thought then a final line; their parser takes the last bare
   yes/no line.
6. **Retrieval depth.** Published at top_200 (and top_50); the library default
   is 20 with threshold 0.1. Arm A uses the library defaults deliberately: the
   paper characterises systems as shipped.
7. **Ingestion granularity.** Pairs (2 messages per `add`) versus whole
   sessions. Extraction context differs, so stored facts differ.
8. **Hybrid retrieval prerequisites.** spaCy `en_core_web_sm` and `fastembed`
   must be installed for entity boosting and BM25. Their image installs
   spaCy; ours records both flags in the fingerprint. In this sandbox both
   are absent (no network), so any local run is semantic-only and is labelled
   as such.
9. **Model stack.** "Production-representative model stack" is not named for
   the platform result. Arm B uses their repo's OSS defaults (gpt-4o-mini
   extraction, text-embedding-3-small, gpt-4o answer/judge).
10. **Judge noise.** Vendor: ±1 point. Our tolerance already exceeds it.

## 5. Protocol

```bash
# 0. one-time
python scripts/fetch_longmemeval.py            # record sha256 into both benchmark configs (expected_sha256)
docker compose -f compose/mem0/docker-compose.yml up -d
pip install "mem0ai[nlp]==2.0.20" fastembed && python -m spacy download en_core_web_sm
python -m proxy --port 8811 --upstream https://api.openai.com/v1 --log results/raw/proxy/m4.jsonl --require-attribution

# 1. Arm B (reproduction), all 500 instances
python -m bench.run --system mem0 --benchmark longmemeval_s --seed 42 \
    --config configs/mem0-published-protocol.yaml \
    --benchmark-config configs/benchmarks/longmemeval_s_mem0protocol.yaml

# 2. Arm A (official protocol), three seeds
for s in 42 43 44; do python -m bench.run --system mem0 --benchmark longmemeval_s --seed $s --config configs/mem0.yaml; done

# 3. report (analysis only; no system is run)
python scripts/reproduce_mem0.py --run results/raw/runs/<B> --run results/raw/runs/<A42> --run results/raw/runs/<A43> --run results/raw/runs/<A44> --proxy-log results/raw/proxy/m4.jsonl
```

Before step 1: stop every other system's containers, verify only
`memharness-mem0-qdrant` is running, `reset_all()` is performed by the runner
per instance via `reset(question_id)` and the collection is fresh
(`docker compose ... down -v` before the campaign), check `/healthz` on the
proxy, and confirm `overrides_present: false` in the manifest.

## 6. What this milestone can and cannot conclude

* If Arm B is **reproduced**: the OSS default stack matches the platform
  number under Mem0's protocol; the paper reports Arm A as Mem0's
  official-protocol accuracy and the A–B gap as the prompt/depth effect.
* If Arm B is **not_reproduced**: the discrepancy is attributed, in order, to
  items 1–3 above (platform-only code paths, unpinned software, and a server
  that does not do what its runner assumes), with item 4 measured directly by
  the A–B gap. That is a valid, documented outcome under the plan's rule that
  a failed reproduction with a documented setup is a result.
* Nothing here may be tuned after seeing Arm A or Arm B.

## 7. Execution status

Not executed. Blockers in the authoring environment: no access to
huggingface.co (dataset), api.openai.com (models), or Docker daemon (Qdrant
server). The apparatus has been validated end to end on a synthetic
LongMemEval-schema dataset with a scripted fake provider
(`scripts/demo_milestone4.py`), which proves the path and the report, not the
number.
