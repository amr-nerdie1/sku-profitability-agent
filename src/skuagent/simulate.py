"""Synthetic order-level data for a large-item furniture marketplace.

Fully invented. Each synthetic SKU is given a hidden ``true_driver`` (why it
loses money, if it does), so we can measure how accurately the pipeline
diagnoses it. The generator also plants the two data traps the real system
had to learn to handle:

1. **Return freight inside the freight column.** High-refund products carry
   return shipping in ``freight_total``, so a naive read calls them freight
   problems.
2. **Buyer remorse that looks like a listing problem.** Remorse returns are
   frequent on cheap, light items; sending them to "fix the listing" wastes
   the content team's time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["Office Desks", "Dining Tables", "Bookcases", "Accent Chairs", "Bar Stools",
              "TV Stands", "Mirrors", "Coffee Tables", "Bed Frames", "Outdoor Planters"]
MATERIALS = ["wood", "metal", "glass", "stone", "upholstered"]
CARRIERS_PARCEL = ["ParcelCo", "QuickShip"]
CARRIERS_LTL = ["FreightOne", "BigHaul", "LineLTL"]
REASONS = ["buyer_remorse", "damage", "defect", "carrier_delay", "listing_error", "lower_price", "unknown"]

DRIVERS = {
    # true driver: share of SKUs
    "healthy": 0.46,
    "high_cogs": 0.12,
    "freight_dims": 0.08,
    "routing_gap": 0.05,
    "carrier_damage": 0.06,
    "buyer_remorse": 0.10,
    "defect": 0.05,
    "listing_error": 0.04,
    "low_roas": 0.04,
}


def simulate(n_skus: int = 3000, n_vendors: int = 45, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (skus, order_lines)."""
    rng = np.random.default_rng(seed)
    vendors = [f"V{i:03d}" for i in range(n_vendors)]
    drv = rng.choice(list(DRIVERS), size=n_skus, p=list(DRIVERS.values()))

    sku = pd.DataFrame({
        "sku": [f"S{i:05d}" for i in range(n_skus)],
        "vendor": rng.choice(vendors, n_skus),
        "category": rng.choice(CATEGORIES, n_skus),
        "material": rng.choice(MATERIALS, n_skus, p=[.45, .2, .12, .08, .15]),
        "private_label": rng.random(n_skus) < .25,
        "true_driver": drv,
    })
    sku["weight_lb"] = np.round(rng.lognormal(3.6, .7, n_skus), 1)
    sku["price"] = np.round(np.clip(sku.weight_lb * rng.uniform(4, 11, n_skus), 40, 3000), 2)
    sku["cogs_pct"] = np.clip(rng.normal(.58, .05, n_skus), .40, .75)
    sku["ship_mode"] = np.where(sku.weight_lb > 120, "LTL", "parcel")
    sku["est_freight"] = np.round(np.where(sku.ship_mode == "LTL", 95 + .55 * sku.weight_lb, 12 + .45 * sku.weight_lb), 2)
    sku["mkt_per_order"] = np.round(sku.price * rng.uniform(.01, .04, n_skus), 2)
    sku["damage_rate"] = rng.uniform(.01, .04, n_skus)
    sku["refund_rate"] = rng.uniform(.02, .06, n_skus)
    sku["freight_mult"] = rng.normal(1.0, .05, n_skus)
    sku["bad_carrier"] = None

    # plant drivers
    m = sku.true_driver
    sku.loc[m == "high_cogs", "cogs_pct"] = rng.uniform(.80, .92, (m == "high_cogs").sum())
    sku.loc[m == "freight_dims", "freight_mult"] = rng.uniform(1.8, 2.6, (m == "freight_dims").sum())
    rg = m == "routing_gap"
    sku.loc[rg, "material"] = rng.choice(["stone", "glass"], rg.sum())
    sku.loc[rg, "weight_lb"] = rng.uniform(55, 115, rg.sum()).round(1)
    sku.loc[rg, "ship_mode"] = "parcel"
    sku.loc[rg, "est_freight"] = (12 + .45 * sku.loc[rg, "weight_lb"]).round(2)
    sku.loc[rg, "damage_rate"] = rng.uniform(.25, .55, rg.sum())
    cd = m == "carrier_damage"
    sku.loc[cd, "bad_carrier"] = [rng.choice(CARRIERS_LTL if s == "LTL" else CARRIERS_PARCEL) for s in sku.loc[cd, "ship_mode"]]
    br = m == "buyer_remorse"
    sku.loc[br, "refund_rate"] = rng.uniform(.25, .40, br.sum())
    sku.loc[br, "weight_lb"] = rng.uniform(8, 45, br.sum()).round(1)
    sku.loc[br, "ship_mode"] = "parcel"
    sku.loc[br, "est_freight"] = (12 + .45 * sku.loc[br, "weight_lb"]).round(2)
    sku.loc[m.isin(["defect", "listing_error"]), "refund_rate"] = rng.uniform(.18, .32, m.isin(["defect", "listing_error"]).sum())
    sku.loc[m == "low_roas", "mkt_per_order"] = (sku.loc[m == "low_roas", "price"] * rng.uniform(.25, .40, (m == "low_roas").sum())).round(2)

    # order volume: heavy tail, some thin SKUs (drives confidence tiers)
    n_orders = np.clip(np.round(rng.lognormal(3.0, 1.0, n_skus)), 1, 600).astype(int)
    rows = []
    for i, r in sku.iterrows():
        k = n_orders[i]
        carriers = CARRIERS_LTL if r.ship_mode == "LTL" else CARRIERS_PARCEL
        car = rng.choice(carriers, k)
        dmg_p = np.full(k, r.damage_rate)
        if r.bad_carrier:
            dmg_p = np.where(car == r.bad_carrier, .35, dmg_p)
        damaged = rng.random(k) < dmg_p
        base_ref = rng.random(k) < r.refund_rate
        refunded = base_ref | (damaged & (rng.random(k) < .8))
        reason = np.full(k, None, dtype=object)
        reason[damaged & refunded] = "damage"
        other = refunded & ~damaged
        if r.true_driver == "buyer_remorse":
            p = [.75, 0, .03, .04, .03, .10, .05]
        elif r.true_driver == "defect":
            p = [.10, 0, .70, .03, .05, .02, .10]
        elif r.true_driver == "listing_error":
            p = [.12, 0, .06, .03, .70, .02, .07]
        else:
            p = [.35, 0, .15, .15, .08, .12, .15]
        reason[other] = rng.choice(REASONS, other.sum(), p=p)
        # some SKUs never captured return reasons
        if rng.random() < .07:
            reason[refunded] = "unknown"
        out_freight = r.est_freight * r.freight_mult * rng.normal(1, .06, k)
        # TRAP 1: return shipping lands in the same freight column
        ret_freight = np.where(refunded & np.isin(reason, ["buyer_remorse", "defect", "listing_error", "lower_price"]),
                               r.est_freight * rng.uniform(.8, 1.2, k), 0.0)
        rows.append(pd.DataFrame({
            "order_id": [f"O{i:05d}-{j:03d}" for j in range(k)],
            "sku": r.sku,
            "carrier": car,
            "gross": r.price,
            "cogs": r.price * r.cogs_pct,
            "est_freight": r.est_freight,
            "freight_total": np.round(out_freight + ret_freight, 2),
            "return_freight": np.round(ret_freight, 2),
            "refund": np.where(refunded, r.price * rng.uniform(.85, 1.0, k), 0.0).round(2),
            "refund_reason": reason,
            "damaged": damaged,
            "marketing": r.mkt_per_order,
            "ad_sales": np.where(rng.random(k) < .3, r.price, 0.0),
        }))
    orders = pd.concat(rows, ignore_index=True)
    sku = sku.drop(columns=["damage_rate", "refund_rate", "freight_mult", "mkt_per_order"])
    return sku, orders


if __name__ == "__main__":
    s, o = simulate()
    print(s.true_driver.value_counts(), len(o))
