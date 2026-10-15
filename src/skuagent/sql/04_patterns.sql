-- Layer 4: vendor and category patterns, so one action can cover many SKUs.
CREATE OR REPLACE TABLE l4_vendor AS
SELECT s.vendor,
       SUM(r.cogs) / NULLIF(SUM(r.gross), 0)                AS vendor_cogs_pct,
       SUM(r.qty_damaged) * 1.0 / NULLIF(SUM(r.orders), 0)  AS vendor_dmg_rate,
       SUM(r.reason_defect) * 1.0 / NULLIF(SUM(r.orders), 0) AS vendor_defect_rate
FROM l2_sku r JOIN skus s USING (sku) GROUP BY s.vendor;

CREATE OR REPLACE TABLE l4_category AS
SELECT s.category,
       SUM(r.outbound_freight) / NULLIF(SUM(r.est_freight), 0) AS cat_freight_ratio,
       SUM(r.reason_buyer_remorse) * 1.0 / NULLIF(SUM(r.orders), 0) AS cat_remorse_rate
FROM l2_sku r JOIN skus s USING (sku) GROUP BY s.category;
