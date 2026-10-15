"""The agent: Claude drives a weekly loop over deterministic tools.

    run_pipeline  ->  get_action_summary  ->  inspect_sku (as needed)
                  ->  validate_sample (LLM-as-judge)  ->  create_jira_ticket (per action)
                  ->  write_report (weekly insights in plain English)

Guardrails are in the tools, not the prompt:
* numbers only come from the SQL layers; the model cannot invent a figure
* Low-confidence SKUs never get an individual ticket
* an action group is only ticketed if the judge agrees with at least 70% of its sampled SKUs;
  otherwise it is held for human review
* Jira tickets are dry-run unless Jira credentials are configured

With ANTHROPIC_API_KEY set, Claude chooses the order of tool calls and writes the insights.
Without it, ``run_offline`` executes the same tools in a fixed plan so anyone can run the demo.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .jira import JiraClient, build_tickets
from .judge import judge, stratified_sample
from .pipeline import run_pipeline
from .report import write_report
from .taxonomy import ACTIONS, diagnose

AGREEMENT_GATE = 0.70

SYSTEM = """You are the SKU profitability agent for a large-item furniture marketplace.
Each week you: run the pipeline, review the action summary, validate a sample with the judge,
raise one Jira ticket per action that is ready, and write a short plain-English insights note
for leadership (what changed, where the money is, what each team must do).
Rules: use only numbers returned by tools. Never ticket an action the validation tool marks as HOLD.
Never ticket SPA-900 or SPA-999 individually. Keep the insights under 200 words."""

TOOLS = [
    {"name": "run_pipeline", "description": "Run the SQL layers and taxonomy for this week's data. Returns headline stats.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_action_summary", "description": "Action groups with SKU counts, $ margin gap, owner, confidence mix.",
     "input_schema": {"type": "object", "properties": {"top_n": {"type": "integer"}}}},
    {"name": "inspect_sku", "description": "Full evidence for one SKU.",
     "input_schema": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}},
    {"name": "validate_sample", "description": "LLM-as-judge on a stratified sample. Returns agreement per action and READY/HOLD.",
     "input_schema": {"type": "object", "properties": {"per_driver": {"type": "integer"}}}},
    {"name": "create_jira_ticket", "description": "Create the Jira ticket for one action group (owner team's project).",
     "input_schema": {"type": "object", "properties": {"action_id": {"type": "string"}}, "required": ["action_id"]}},
    {"name": "write_report", "description": "Write the weekly Excel workbook with your insights note.",
     "input_schema": {"type": "object", "properties": {"insights": {"type": "string"}}, "required": ["insights"]}},
]


class Agent:
    def __init__(self, skus: pd.DataFrame, orders: pd.DataFrame, week: str, out_dir: str | Path = "output"):
        self.skus, self.orders, self.week = skus, orders, week
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        tdir = self.out / "tickets"
        if tdir.exists():
            for p in tdir.glob("*"):
                p.unlink()
        self.jira = JiraClient(tdir)
        self.diag = None
        self.validation = None
        self.gate = {}
        self.tickets = []
        self.log = []

    # ---------------------------------------------------------------- tools
    def run_pipeline(self):
        f = run_pipeline(self.skus, self.orders)
        self.diag = diagnose(f)
        d = self.diag
        neg = d[d.contribution < 0]
        return {"skus": len(d), "gross": round(d.gross.sum()), "contribution": round(d.contribution.sum()),
                "cm_pct": round(d.contribution.sum() / d.gross.sum(), 4), "negative_cm_skus": len(neg),
                "loss_on_negative_cm_skus": round(neg.contribution.sum()),
                "drivers": d.primary_driver.value_counts().to_dict(), "confidence": d.confidence.value_counts().to_dict()}

    def action_summary(self) -> pd.DataFrame:
        a = self.diag[self.diag.action_id.notna()]
        s = a.groupby(["primary_driver", "action_id"]).agg(
            skus=("sku", "count"), gross=("gross", "sum"), contribution=("contribution", "sum"),
            margin_gap=("margin_gap", "sum"), high=("confidence", lambda c: (c == "High").sum()),
            medium=("confidence", lambda c: (c == "Medium").sum()), low=("confidence", lambda c: (c == "Low").sum())).reset_index()
        s["action"] = s.action_id.map(lambda x: ACTIONS[x].description)
        s["owner"] = s.action_id.map(lambda x: ACTIONS[x].owner)
        s["fixability"] = s.action_id.map(lambda x: ACTIONS[x].fixability)
        return s.sort_values("margin_gap", ascending=False)

    def get_action_summary(self, top_n: int = 12):
        s = self.action_summary().head(top_n)
        return s[["action_id", "action", "owner", "skus", "margin_gap", "contribution", "high", "medium", "low", "fixability"]].round(0).to_dict("records")

    def inspect_sku(self, sku: str):
        r = self.diag.loc[self.diag.sku == sku]
        if r.empty:
            return {"error": "unknown sku"}
        r = r.iloc[0]
        return {k: (r[k] if not isinstance(r[k], float) else round(r[k], 4)) for k in
                ["sku", "vendor", "category", "primary_driver", "root_cause", "action_id", "owner", "confidence", "evidence", "margin_gap"]}

    def validate_sample(self, per_driver: int = 6):
        self.validation = judge(stratified_sample(self.diag, per_driver))
        by = self.validation.groupby("rules_action").agg(n=("sku", "count"), agree=("action_agree", "mean"))
        by["status"] = by.agree.map(lambda x: "READY" if x >= AGREEMENT_GATE else "HOLD")
        self.gate = by.status.to_dict()
        return {"mode": self.validation.judge_mode.iloc[0], "sampled": len(self.validation),
                "driver_agreement": round(float(self.validation.driver_agree.mean()), 3),
                "action_agreement": round(float(self.validation.action_agree.mean()), 3),
                "by_action": by.round(2).reset_index().to_dict("records")}

    def create_jira_ticket(self, action_id: str):
        if action_id in ("SPA-900", "SPA-999"):
            return {"refused": "pattern-level / manual-review actions are not ticketed individually"}
        if self.gate.get(action_id) == "HOLD":
            return {"refused": f"{action_id} held: judge agreement below {AGREEMENT_GATE:.0%}; route to human review"}
        hi_med = self.diag[self.diag.confidence != "Low"]
        t = build_tickets(hi_med, self.week, actions=[action_id])
        if not t:
            return {"refused": "no High/Medium-confidence SKUs for this action"}
        key = self.jira.create(t[0])
        self.tickets.append({**{k: t[0][k] for k in ["project", "action_id", "owner", "priority", "summary", "margin_gap"]},
                             "key": key, "skus": len(t[0]["skus"]), "mode": "live" if self.jira.live else "dry-run"})
        return self.tickets[-1]

    def write_report(self, insights: str):
        path = write_report(self.out / f"sku_profitability_{self.week}.xlsx", self.diag, self.action_summary(),
                            self.validation, pd.DataFrame(self.tickets), insights)
        (self.out / f"insights_{self.week}.md").write_text(insights)
        return {"path": str(path)}

    # ---------------------------------------------------------------- loops
    def _call(self, name, args):
        out = getattr(self, name)(**(args or {}))
        self.log.append({"tool": name, "input": args, "output": out})
        return out

    def run_live(self, model: str | None = None, max_turns: int = 30):
        import anthropic
        client = anthropic.Anthropic()
        model = model or os.environ.get("SKUAGENT_MODEL", "claude-sonnet-4-5")
        messages = [{"role": "user", "content": f"Run the weekly SKU profitability cycle for week {self.week}."}]
        for _ in range(max_turns):
            resp = client.messages.create(model=model, max_tokens=2000, system=SYSTEM, tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    out = self._call(block.name, block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out, default=str)})
            messages.append({"role": "user", "content": results})
        return self.log

    def run_offline(self):
        """Same tools, fixed plan, templated insights. No API key needed."""
        head = self._call("run_pipeline", {})
        summary = self._call("get_action_summary", {"top_n": 20})
        val = self._call("validate_sample", {"per_driver": 8})
        for row in summary:
            if row["action_id"] in ("SPA-900", "SPA-999") or row["high"] + row["medium"] == 0:
                continue
            self._call("create_jira_ticket", {"action_id": row["action_id"]})
        held = [a for a, s in self.gate.items() if s == "HOLD"]
        top3 = [t for t in self.tickets][:3]
        insights = "\n".join([
            f"Week {self.week}: {head['skus']:,} SKUs, contribution margin {head['cm_pct']:.1%}.",
            f"{head['negative_cm_skus']:,} SKUs lose money on every sale, ${-head['loss_on_negative_cm_skus']:,.0f} in total.",
            "Biggest opportunities this week:",
            *[f"- {t['owner']}: {t['summary']} ({t['key']})" for t in top3],
            f"{len(self.tickets)} Jira tickets raised across {len({t['owner'] for t in self.tickets})} teams.",
            f"Validation ({val['mode']}): judge agreed on {val['action_agreement']:.0%} of sampled actions."
            + (f" Held for human review: {', '.join(held)}." if held else ""),
            f"{head['confidence'].get('Low', 0):,} thin SKUs are monitored at pattern level only.",
        ])
        self._call("write_report", {"insights": insights})
        return self.log
