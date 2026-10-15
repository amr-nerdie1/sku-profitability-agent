-- Layer 1: order-line economics.
-- ACCURACY FIX #1: return shipping is booked in the same freight column as
-- outbound freight. Unless it is separated, high-refund SKUs look like
-- freight problems. {SEPARATE_RETURN_FREIGHT} toggles the fix (TRUE in production).
CREATE OR REPLACE TABLE l1_order_economics AS
SELECT
    o.order_id, o.sku, o.carrier, o.gross, o.cogs, o.est_freight,
    CASE WHEN {SEPARATE_RETURN_FREIGHT} THEN o.freight_total - o.return_freight
         ELSE o.freight_total END                       AS outbound_freight,
    CASE WHEN {SEPARATE_RETURN_FREIGHT} THEN o.return_freight ELSE 0 END AS return_freight,
    o.refund, o.refund_reason, o.damaged, o.marketing, o.ad_sales
FROM orders o;
