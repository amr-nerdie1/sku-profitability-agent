"""Turn action groups into Jira tickets for the owning team.

Live: set JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN (and optionally JIRA_ISSUE_TYPE) and
tickets are created with the Jira Cloud REST API v3.
Dry run (default): each ticket is written to output/tickets/<KEY>.json plus a
readable output/tickets/TICKETS.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .taxonomy import ACTIONS

ACCEPTANCE = {
    "SPA-101": "New price floor set per SKU; refund-adjusted CM at or above target after 4 weeks.",
    "SPA-102": "SKUs deactivated or liquidated; no new POs raised.",
    "SPA-104": "Vendor acknowledges defect rate; corrective action or chargeback agreed.",
    "SPA-105": "Competitor prices checked; reprice or hold decision logged per SKU.",
    "SPA-107": "Return reasons captured for >= 80% of refunded orders; SKUs re-run next cycle.",
    "SPA-109": "Carrier SLA breach raised; late-delivery return rate tracked weekly.",
    "SPA-201": "Listing content corrected and approved; listing-error returns tracked for 4 weeks.",
    "SPA-301": "Ship mode switched to LTL in OMS; damage rate re-measured after 30 orders.",
    "SPA-302": "Carrier excluded for these SKUs in routing rules; damage rate re-measured.",
    "SPA-303": "Packaging spec updated with vendor; damage rate re-measured after 30 orders.",
    "SPA-401": "Dims and weight verified and corrected in product master.",
    "SPA-402": "Category freight pre-estimate recalibrated; variance under 10% next cycle.",
    "SPA-501": "Price raised to margin target or SKU flagged for exit with rationale.",
    "SPA-502": "Cost renegotiation opened with vendor; target COGS documented.",
    "SPA-503": "Exit decision approved; inventory plan agreed.",
    "SPA-601": "Targeting tightened or campaign paused; ROAS at or above 3 after 2 weeks.",
    "SPA-900": "Pattern-level review only: no individual SKU action.",
}


def build_tickets(diag: pd.DataFrame, week: str, actions: list[str] | None = None, top_n_skus: int = 15) -> list[dict]:
    act = diag[diag.action_id.notna() & (diag.action_id != "SPA-999")]
    groups = act.groupby("action_id")
    tickets = []
    for aid, g in groups:
        if actions and aid not in actions:
            continue
        a = ACTIONS[aid]
        gap = g.margin_gap.sum()
        loss = g.loc[g.contribution < 0, "contribution"].sum()
        top = g.sort_values("margin_gap", ascending=False).head(top_n_skus)
        lines = [f"| {r.sku} | {r.vendor} | {r.category} | {r.cont_pct:.1%} | ${r.margin_gap:,.0f} | {r.confidence} | {r.root_cause} |"
                 for r in top.itertuples()]
        desc = "\n".join([
            f"*Action {aid}: {a.description}*", "",
            f"Raised automatically by the SKU profitability agent, week {week}.",
            f"{len(g)} SKUs. Margin gap to target: ${gap:,.0f}. Current loss on negative-CM SKUs: ${-loss:,.0f}.",
            f"Confidence mix: {g.confidence.value_counts().to_dict()}.", "",
            "| SKU | Vendor | Category | CM % | Gap $ | Confidence | Root cause |",
            "|---|---|---|---|---|---|---|", *lines,
            "" if len(g) <= top_n_skus else f"...and {len(g) - top_n_skus} more SKUs in the attached report.",
            "", f"*Acceptance criteria:* {ACCEPTANCE.get(aid, '')}",
            "*Evidence:* every SKU row in the weekly workbook carries its arithmetic in words.",
        ])
        priority = "Highest" if gap >= 250_000 else "High" if gap >= 75_000 else "Medium" if gap >= 15_000 else "Low"
        tickets.append({"project": a.jira_project, "action_id": aid, "owner": a.owner, "priority": priority,
                        "summary": f"[{aid}] {a.description}: {len(g)} SKUs, ${gap:,.0f} margin gap",
                        "description": desc, "labels": ["sku-profitability", f"wk-{week}", aid.lower()],
                        "skus": g.sku.tolist(), "margin_gap": round(float(gap), 2)})
    return sorted(tickets, key=lambda t: -t["margin_gap"])


class JiraClient:
    def __init__(self, out_dir: str | Path = "output/tickets"):
        self.url = os.environ.get("JIRA_URL")
        self.auth = (os.environ.get("JIRA_EMAIL"), os.environ.get("JIRA_API_TOKEN"))
        self.issue_type = os.environ.get("JIRA_ISSUE_TYPE", "Task")
        self.live = bool(self.url and all(self.auth))
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._n = {}

    def create(self, t: dict) -> str:
        if self.live:
            import requests
            body = {"fields": {
                "project": {"key": t["project"]}, "issuetype": {"name": self.issue_type},
                "summary": t["summary"], "labels": t["labels"], "priority": {"name": t["priority"]},
                "description": {"type": "doc", "version": 1, "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": line or " "}]}
                    for line in t["description"].split("\n")]},
            }}
            r = requests.post(f"{self.url}/rest/api/3/issue", json=body, auth=self.auth, timeout=30)
            r.raise_for_status()
            return r.json()["key"]
        self._n[t["project"]] = self._n.get(t["project"], 0) + 1
        key = f"{t['project']}-{100 + self._n[t['project']]}"
        (self.out / f"{key}.json").write_text(json.dumps({**t, "key": key}, indent=2))
        with open(self.out / "TICKETS.md", "a") as fh:
            fh.write(f"\n## {key} · {t['priority']} · {t['owner']}\n**{t['summary']}**\n\n{t['description']}\n\n---\n")
        return key
