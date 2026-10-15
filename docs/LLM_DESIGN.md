# LLM design: prompts, guardrails, hallucination control

How this agent keeps an LLM from making things up in a report that leadership acts on. The full cross-repo write-up, with the red-team eval suite, is in [category-growth-agent/docs/LLM_DESIGN.md](https://github.com/amr-nerdie1/category-growth-agent/blob/main/docs/LLM_DESIGN.md).

## Where the LLM is used, and where it isn't

| Step | Who does it | Why |
|---|---|---|
| Margin, $ gap, driver, action ID, owner, priority | SQL + Python (`pipeline.py`, `taxonomy.py`) | Numbers and routing decisions must be reproducible |
| Choosing which tools to call, which SKUs to inspect | LLM (agent loop) | Judgement about where to look |
| Second opinion on the diagnosis | LLM-as-judge (`judge.py`) | Catches rule errors on a stratified sample |
| Weekly insights note for leadership | LLM, then **checked** | Plain-English narrative |
| Jira ticket text | Code templates (`jira.py`) | Tickets are instructions to teams, so they're never free text |

## Prompts (verbatim in code)

* **Agent system prompt:** `SYSTEM` in [`agent.py`](../src/skuagent/agent.py). It sets 4 numbered rules: only tool numbers, only tool-returned IDs, never ticket HOLD / SPA-900 / SPA-999, and a 200-word limit. It also says what to do when `write_report` rejects the note.
* **Judge prompt:** `PROMPT` in [`judge.py`](../src/skuagent/judge.py). The judge sees **only the raw evidence and the action menu, not the rules engine's answer**, so it can't simply agree. It must reply as JSON (`driver`, `action_id`, one-sentence `reason`); unparseable replies count as disagreement.

## Guardrails

| Risk | Guardrail | Code |
|---|---|---|
| Invented $ or % in the insights note | Every number must match a tool result (unit-aware: `$4.5M` needs a value near 4,500,000) | `check_insights` → `guardrails.verify_numbers` |
| Invented action ID, Jira key or SKU | Must exist in `ACTIONS`, in tickets actually created this run, or in the SKU table | `check_insights` |
| Model ignores the rejection | `write_report` refuses twice with the exact problems. The third failure publishes the deterministic template note, marked "model-written note withheld" | `write_report`, `MAX_REPAIRS = 2` |
| Rules engine is wrong for a whole action group | Judge must agree on at least 70% of a sampled group, or that group's ticket is held for human review | `AGREEMENT_GATE`, `create_jira_ticket` |
| Acting on thin data | SKUs with fewer than 10 orders never get an individual action or ticket | `taxonomy.py` confidence tiers |
| Running totals the model will report | Ticket tool returns `tickets_so_far` / `teams_so_far`, so counts are grounded too | `create_jira_ticket` |
| Side effects | Jira is a dry run unless credentials are set; tickets are built by code, not the model | `jira.py` |

There is no third-party free text in this pipeline (all inputs are internal order data), so prompt injection is a lower risk here than in [ltl-freight-auditor](https://github.com/amr-nerdie1/ltl-freight-auditor), where carrier emails are the main input.

## Tests

`pytest -q` runs 13 tests. Six cover the hallucination guardrails:

* the template note passes grounding;
* five fabricated sentences are each rejected: `$4.5M` EBITDA, `14.8%` margin, `SPA-777`, `PRC-999`, `S99999`;
* a model that keeps fabricating gets two rejections, then its note is withheld.
