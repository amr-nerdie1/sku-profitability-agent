"""LLM-as-judge: independently re-check a stratified sample of diagnoses before anything reaches leadership.

Live mode (ANTHROPIC_API_KEY set): Claude sees only the raw evidence for a SKU and the
action menu, picks a driver and action itself, and we compare it with the rules.

Offline mode: a reference judge built from a *different* rule path (cost-share
ranking rather than $-drag ranking) so the demo runs without a key. It is labelled as
such in every output.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .taxonomy import ACTIONS

DRIVER_FAMILY = {"REFUND": {"SPA-101", "SPA-102", "SPA-104", "SPA-105", "SPA-107", "SPA-109", "SPA-201"},
                 "DAMAGE": {"SPA-301", "SPA-302", "SPA-303"}, "FREIGHT": {"SPA-401", "SPA-402"},
                 "COGS": {"SPA-501", "SPA-502", "SPA-503"}, "MARKETING": {"SPA-601"}}

PROMPT = """You are auditing a SKU profitability diagnosis for a furniture marketplace.
Using ONLY the evidence below, decide the primary driver of lost margin and the single best action.

Evidence:
{evidence}
Refund $ by reason: {reasons}
Routing: {material}, {weight} lb, ships {mode}. Top-damage carrier lift: {lift}
Vendor COGS: {vcogs}. Category freight vs estimate: {catfr}

Drivers: REFUND, DAMAGE, FREIGHT, COGS, MARKETING.
Actions: {menu}

Reply with JSON only: {{"driver": "...", "action_id": "SPA-...", "reason": "one sentence"}}"""


def stratified_sample(diag: pd.DataFrame, per_driver: int = 6, seed: int = 0) -> pd.DataFrame:
    pool = diag[(diag.confidence != "Low") & diag.primary_driver.isin(DRIVER_FAMILY)]
    parts = [g.sample(min(per_driver, len(g)), random_state=seed) for _, g in pool.groupby("primary_driver")]
    return pd.concat(parts)


def _prompt(r) -> str:
    reasons = {k: round(r[f"refund_{k}"]) for k in ["buyer_remorse", "damage", "defect", "carrier_delay", "listing_error", "lower_price", "unknown"]}
    menu = "; ".join(f"{a.id} {a.description}" for a in ACTIONS.values() if a.id not in ("SPA-900", "SPA-999"))
    return PROMPT.format(evidence=r.evidence, reasons=reasons, material=r.material, weight=r.weight_lb, mode=r.ship_mode,
                         lift=f"{r.top_damage_carrier} {r.dmg_lift:.1f}x" if pd.notna(r.dmg_lift) else "n/a",
                         vcogs=f"{r.vendor_cogs_pct:.0%}", catfr=f"{r.cat_freight_ratio:.2f}x", menu=menu)


def _offline_judge(r) -> dict:
    """Independent re-derivation: cost-share ranking for the driver, count-based reasons and
    different thresholds for the action. Disagreements are informative, not errors."""
    shares = {"COGS": r.cogs_pct - 0.60, "FREIGHT": (r.freight_vs_estimate or 1) - 1.0,
              "REFUND": (r.refunds - r.refund_damage + r.return_freight) / r.gross - 0.04,
              "DAMAGE": r.refund_damage / r.gross - 0.02,
              "MARKETING": (r.marketing / r.gross - 0.04) if (r.roas or 0) < 3 else -1}
    drv = max(shares, key=shares.get)
    if drv == "REFUND":
        counts = {k: r[f"reason_{k}"] for k in ["buyer_remorse", "defect", "carrier_delay", "listing_error", "lower_price"]}
        dom = max(counts, key=counts.get)
        if (r.unknown_reason_share or 0) >= 0.4:
            act = "SPA-107"
        elif dom == "buyer_remorse":
            act = "SPA-102" if r.cont_pct < -0.30 else "SPA-101"
        else:
            act = {"defect": "SPA-104", "carrier_delay": "SPA-109", "listing_error": "SPA-201", "lower_price": "SPA-105"}[dom]
    elif drv == "DAMAGE":
        if r.material in ("stone", "glass") and r.weight_lb >= 50 and r.ship_mode == "parcel":
            act = "SPA-301"
        elif (r.dmg_lift or 0) >= 2.0:
            act = "SPA-302"
        else:
            act = "SPA-303"
    elif drv == "FREIGHT":
        act = "SPA-402" if (r.cat_freight_ratio or 1) >= 1.2 else "SPA-401"
    elif drv == "COGS":
        act = "SPA-503" if r.cont_pct < -0.20 else "SPA-502" if (r.vendor_cogs_pct or 0) >= 0.72 else "SPA-501"
    else:
        act = "SPA-601"
    return {"driver": drv, "action_id": act, "reason": f"offline reference judge: {drv} by cost share"}


def judge(sample: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    live = bool(os.environ.get("ANTHROPIC_API_KEY"))
    client = None
    if live:
        import anthropic
        client = anthropic.Anthropic()
        model = model or os.environ.get("SKUAGENT_MODEL", "claude-sonnet-4-5")
    rows = []
    for _, r in sample.iterrows():
        if live:
            msg = client.messages.create(model=model, max_tokens=300, messages=[{"role": "user", "content": _prompt(r)}])
            txt = msg.content[0].text
            try:
                v = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            except Exception:
                v = {"driver": "?", "action_id": "?", "reason": txt[:200]}
        else:
            v = _offline_judge(r)
        rows.append({"sku": r.sku, "rules_driver": r.primary_driver, "rules_action": r.action_id,
                     "judge_driver": v.get("driver"), "judge_action": v.get("action_id"),
                     "driver_agree": v.get("driver") == r.primary_driver,
                     "action_agree": v.get("action_id") == r.action_id,
                     "judge_reason": v.get("reason"), "judge_mode": "claude" if live else "offline-reference"})
    return pd.DataFrame(rows)
