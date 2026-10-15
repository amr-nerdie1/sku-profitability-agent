-- Layer 5: heavy, brittle products shipped parcel with high damage and no LTL history.
CREATE OR REPLACE TABLE l5_routing_gap AS
SELECT r.sku, TRUE AS routing_gap
FROM l2_sku r JOIN skus s USING (sku)
WHERE s.ship_mode = 'parcel' AND s.weight_lb >= 50
  AND s.material IN ('stone', 'glass')
  AND r.qty_damaged * 1.0 / r.orders >= 0.15;
