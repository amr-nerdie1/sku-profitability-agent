"""Driver → root cause → action ID → owner → fixability.

The decision matrix is the product: it turns "this SKU loses money" into a
specific action for a specific team. Rules are deterministic and readable, so
every verdict can be traced and challenged. The LLM layer comments on these
outputs and checks them; it never replaces them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- action matrix
@dataclass(frozen=True)
class Action:
    id: str
    description: str
    owner: str          # team that acts
    jira_project: str   # where the ticket goes
    fixability: str     # High / Medium / Low


ACTIONS = {a.id: a for a in [
    Action("SPA-101", "Price floor review: absorb buyer-remorse cost in price", "Pricing", "PRC", "Medium"),
    Action("SPA-102", "Exit SKU: structural remorse, unrecoverable", "Assortment", "PRC", "Low"),
    Action("SPA-104", "Vendor QA escalation: defect returns", "Vendor Management", "VND", "Medium"),
    Action("SPA-105", "Competitive price audit", "Pricing", "PRC", "Medium"),
    Action("SPA-107", "Pull return reasons: cannot diagnose yet", "Customer Service", "CS", "Low"),
    Action("SPA-109", "Carrier SLA review: late-delivery returns", "Logistics", "OPS", "Medium"),
    Action("SPA-201", "Listing correction: content mismatch", "Content", "CNT", "High"),
    Action("SPA-301", "Move to LTL: heavy brittle item damaged on parcel", "Logistics", "OPS", "High"),
    Action("SPA-302", "Stop tendering this SKU x carrier combination", "Logistics", "OPS", "High"),
    Action("SPA-303", "Packaging upgrade: fragile / damage-prone", "Operations", "OPS", "High"),
    Action("SPA-401", "Correct dims / weight master data", "Pricing", "PRC", "High"),
    Action("SPA-402", "Recalibrate category freight pre-estimate", "Pricing", "PRC", "Medium"),
    Action("SPA-501", "Reprice to margin target", "Pricing", "PRC", "Medium"),
    Action("SPA-502", "Vendor cost renegotiation: systemic COGS", "Vendor Management", "VND", "Medium"),
    Action("SPA-503", "Exit SKU: cost structure unworkable", "Assortment", "PRC", "Low"),
    Action("SPA-601", "Pause / tighten ad targeting: low ROAS", "Marketing", "MKT", "High"),
    Action("SPA-900", "Monitor: too few orders to act individually", "Analytics", "ANA", "Low"),
    Action("SPA-999", "Multi-factor: manual review required", "Analytics", "ANA", "Low"),
]}

# benchmarks used to size each driver's $ drag
COGS_BENCH = 0.60          # COGS share above this is drag
REFUND_BENCH = 0.04        # normal non-damage refund share of gross
DAMAGE_BENCH = 0.02        # normal damage refund share of gross
MKT_BENCH = 0.04           # normal marketing share of gross
TARGET_CM = {True: 0.12, False: 0.10}   # private label vs branded


def _drags(f: pd.DataFrame) -> pd.DataFrame:
    nondmg_refund = f.refunds - f.refund_damage
    d = pd.DataFrame(index=f.index)
    d["COGS"] = (f.cogs - COGS_BENCH * f.gross).clip(lower=0)
    d["FREIGHT"] = (f.outbound_freight - f.est_freight).clip(lower=0)
    d["REFUND"] = (nondmg_refund + f.return_freight - REFUND_BENCH * f.gross).clip(lower=0)
    d["DAMAGE"] = (f.refund_damage - DAMAGE_BENCH * f.gross).clip(lower=0)
    d["MARKETING"] = np.where(f.roas.fillna(0) < 3, (f.marketing - MKT_BENCH * f.gross).clip(lower=0), 0)
    return d


def diagnose(features: pd.DataFrame, remorse_routing: bool = True) -> pd.DataFrame:
    """Assign PrimaryDriver, RootCause, ActionID, Owner, Fixability, Confidence and a plain-English Evidence string.

    remorse_routing: ACCURACY FIX #2. Without it, every non-damage refund problem is
    routed to "fix the listing", which sends buyer-remorse SKUs to a team that can't fix them.
    """
    f = features.copy()
    drags = _drags(f)
    f[[f"drag_{c.lower()}" for c in drags.columns]] = drags.values
    top = drags.idxmax(axis=1)
    top_val = drags.max(axis=1)
    second_val = drags.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
    target = f.private_label.map(TARGET_CM)

    drivers, causes, actions = [], [], []
    for i, r in f.iterrows():
        if r.cont_pct >= target[i] or top_val[i] < 0.02 * r.gross:
            drivers.append("HEALTHY"); causes.append("At or above margin target"); actions.append(None); continue
        drv = top[i]
        if second_val[i] > 0.85 * top_val[i]:
            drivers.append("REVIEW"); causes.append(f"No dominant driver ({drv} vs runner-up within 15%)"); actions.append("SPA-999"); continue
        cause, act = _root_cause(drv, r, remorse_routing)
        drivers.append(drv); causes.append(cause); actions.append(act)

    f["primary_driver"], f["root_cause"], f["action_id"] = drivers, causes, actions
    # thin SKUs never get an individual action: they roll into pattern-level monitoring
    f["action_id"] = f.action_id.astype(object).where(f.action_id.notna(), None)
    low = (f.confidence == "Low") & f.action_id.notna()
    f.loc[low, "action_id"] = "SPA-900"
    f["action"] = f.action_id.map(lambda a: getattr(ACTIONS[a], "description") if isinstance(a, str) else "")
    f["owner"] = f.action_id.map(lambda a: getattr(ACTIONS[a], "owner") if isinstance(a, str) else "")
    f["jira_project"] = f.action_id.map(lambda a: getattr(ACTIONS[a], "jira_project") if isinstance(a, str) else "")
    f["fixability"] = f.action_id.map(lambda a: getattr(ACTIONS[a], "fixability") if isinstance(a, str) else "")
    f["margin_gap"] = ((target - f.cont_pct).clip(lower=0) * f.gross).round(2)
    f["evidence"] = f.apply(_evidence, axis=1)
    return f


def _root_cause(drv: str, r, remorse_routing: bool) -> tuple[str, str]:
    if drv == "REFUND":
        reasons = {k: r[f"refund_{k}"] for k in ["buyer_remorse", "defect", "carrier_delay", "listing_error", "lower_price"]}
        if (r.unknown_reason_share or 0) >= 0.5:
            return "Return reasons missing: pull before actioning", "SPA-107"
        if not remorse_routing:
            return "High refunds", "SPA-201"            # the original mis-routing
        dom = max(reasons, key=reasons.get)
        if dom == "buyer_remorse":
            light = r.weight_lb < 50
            label = "Buyer remorse (sub-50 lb: near-zero return friction)" if light else "Buyer remorse"
            return label, ("SPA-102" if r.cont_pct < -0.25 else "SPA-101")
        return {
            "defect": ("Product defect", "SPA-104"),
            "carrier_delay": ("Carrier / ops failure", "SPA-109"),
            "listing_error": ("Listing mismatch", "SPA-201"),
            "lower_price": ("Price competition", "SPA-105"),
        }[dom]
    if drv == "DAMAGE":
        if r.routing_gap:
            return f"Routing gap: {r.material} {r.weight_lb:.0f} lb on parcel, {r.dmg_rate:.0%} damaged", "SPA-301"
        if (r.dmg_lift or 0) >= 2.5 and (r.top_carrier_shipped or 0) >= 5:
            return (f"Carrier damage concentration ({r.top_damage_carrier}: {r.top_carrier_dmg_rate:.0%} vs "
                    f"{r.book_dmg_rate:.0%} book, {r.dmg_lift:.1f}x)"), "SPA-302"
        return ("Product fragility (" + r.material + ")" if r.material in ("glass", "stone") else "Packaging weakness"), "SPA-303"
    if drv == "FREIGHT":
        if (r.cat_freight_ratio or 1) >= 1.25:
            return f"Category pre-estimate bias ({r.category}: {r.cat_freight_ratio:.2f}x)", "SPA-402"
        return f"Dims / weight mismatch (actual {r.freight_vs_estimate:.1f}x estimate)", "SPA-401"
    if drv == "COGS":
        if r.cont_pct < -0.25:
            return "Cost structure unworkable", "SPA-503"
        if (r.vendor_cogs_pct or 0) >= 0.75:
            return f"Vendor-wide COGS ({r.vendor}: {r.vendor_cogs_pct:.0%})", "SPA-502"
        return f"SKU-level COGS ({r.cogs_pct:.0%} of price)", "SPA-501"
    if drv == "MARKETING":
        return f"Low ROAS spend (ROAS {r.roas:.1f})", "SPA-601"
    return "Unclassified", "SPA-999"


def _evidence(r) -> str:
    if r.primary_driver == "HEALTHY":
        return f"CM {r.cont_pct:.1%} on {r.orders} orders."
    return (f"{r.orders} orders, CM {r.cont_pct:.1%}. COGS {r.cogs_pct:.0%}, freight {r.freight_pct:.0%} "
            f"({r.freight_vs_estimate:.2f}x est.), refunds {r.refund_pct:.0%} (return freight ${r.return_freight:,.0f}), "
            f"damage {r.dmg_rate:.0%}, ROAS {0 if pd.isna(r.roas) else r.roas:.1f}. "
            f"Largest drag: {r.primary_driver}. Confidence {r.confidence}.")


# ------------------------------------------------------------ evaluation helpers
EXPECTED_DRIVER = {"healthy": "HEALTHY", "high_cogs": "COGS", "freight_dims": "FREIGHT", "routing_gap": "DAMAGE",
                   "carrier_damage": "DAMAGE", "buyer_remorse": "REFUND", "defect": "REFUND",
                   "listing_error": "REFUND", "low_roas": "MARKETING"}
EXPECTED_ACTIONS = {"high_cogs": {"SPA-501", "SPA-502", "SPA-503"}, "freight_dims": {"SPA-401", "SPA-402"},
                    "routing_gap": {"SPA-301"}, "carrier_damage": {"SPA-302"}, "buyer_remorse": {"SPA-101", "SPA-102"},
                    "defect": {"SPA-104"}, "listing_error": {"SPA-201"}, "low_roas": {"SPA-601"}}


def accuracy(diag: pd.DataFrame, skus: pd.DataFrame) -> dict:
    """Driver and action accuracy vs the hidden truth, on SKUs we would act on (High/Medium confidence)."""
    d = diag.merge(skus[["sku", "true_driver"]], on="sku")
    act = d[d.confidence != "Low"]
    loss = act[act.true_driver != "healthy"]
    drv_ok = (loss.primary_driver == loss.true_driver.map(EXPECTED_DRIVER)).mean()
    act_ok = loss.apply(lambda r: r.action_id in EXPECTED_ACTIONS[r.true_driver], axis=1).mean()
    remorse = loss[loss.true_driver == "buyer_remorse"]
    return {
        "driver_accuracy": round(float(drv_ok), 3),
        "action_accuracy": round(float(act_ok), 3),
        "remorse_sent_to_listing_fix": round(float((remorse.action_id == "SPA-201").mean()), 3) if len(remorse) else 0.0,
        "false_freight_flags": int(((d.freight_vs_estimate >= 1.25) & (d.true_driver != "freight_dims")).sum()),
        "n_evaluated": int(len(loss)),
    }
