-- Layer 6: one wide row per SKU with ratios, flags and the evidence the taxonomy needs.
CREATE OR REPLACE TABLE l6_features AS
SELECT
    s.sku, s.vendor, s.category, s.material, s.weight_lb, s.ship_mode, s.private_label, s.price,
    r.*  EXCLUDE (sku),
    r.contribution / NULLIF(r.gross, 0)                      AS cont_pct,
    r.cogs / NULLIF(r.gross, 0)                              AS cogs_pct,
    r.outbound_freight / NULLIF(r.gross, 0)                  AS freight_pct,
    r.outbound_freight / NULLIF(r.est_freight, 0)            AS freight_vs_estimate,
    r.refunds / NULLIF(r.gross, 0)                           AS refund_pct,
    r.qty_damaged * 1.0 / r.orders                           AS dmg_rate,
    r.ad_sales / NULLIF(r.marketing, 0)                      AS roas,
    r.marketing / NULLIF(r.gross, 0)                         AS marketing_pct,
    r.unknown_reason_orders * 1.0 / NULLIF(r.refunded_orders, 0) AS unknown_reason_share,
    c.top_damage_carrier, c.top_carrier_shipped, c.top_carrier_dmg_rate, c.book_dmg_rate, c.dmg_lift,
    v.vendor_cogs_pct, v.vendor_dmg_rate, v.vendor_defect_rate,
    g.cat_freight_ratio, g.cat_remorse_rate,
    COALESCE(rg.routing_gap, FALSE)                          AS routing_gap,
    CASE WHEN r.orders >= 20 AND COALESCE(r.unknown_reason_orders * 1.0 / NULLIF(r.refunded_orders, 0), 0) < 0.5 THEN 'High'
         WHEN r.orders >= 10 THEN 'Medium'
         ELSE 'Low' END                                       AS confidence
FROM l2_sku r
JOIN skus s USING (sku)
LEFT JOIN l3_carrier_lift c USING (sku)
LEFT JOIN l4_vendor v ON v.vendor = s.vendor
LEFT JOIN l4_category g ON g.category = s.category
LEFT JOIN l5_routing_gap rg USING (sku);
