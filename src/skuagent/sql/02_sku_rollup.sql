-- Layer 2: roll up to SKU. Contribution = gross - COGS - freight - refunds - marketing.
CREATE OR REPLACE TABLE l2_sku AS
SELECT
    e.sku,
    COUNT(*)                                   AS orders,
    SUM(gross)                                 AS gross,
    SUM(cogs)                                  AS cogs,
    SUM(outbound_freight)                      AS outbound_freight,
    SUM(return_freight)                        AS return_freight,
    SUM(est_freight)                           AS est_freight,
    SUM(refund)                                AS refunds,
    SUM(marketing)                             AS marketing,
    SUM(ad_sales)                              AS ad_sales,
    SUM(CASE WHEN damaged THEN 1 ELSE 0 END)   AS qty_damaged,
    SUM(CASE WHEN refund > 0 THEN 1 ELSE 0 END) AS refunded_orders,
    SUM(CASE WHEN refund_reason = 'unknown' THEN 1 ELSE 0 END) AS unknown_reason_orders,
    {REASON_COLUMNS}
    SUM(gross) - SUM(cogs) - SUM(outbound_freight) - SUM(return_freight) - SUM(refund) - SUM(marketing) AS contribution
FROM l1_order_economics e
GROUP BY e.sku;
