-- Layer 3: is one carrier damaging this SKU far more than its own book average?
CREATE OR REPLACE TABLE l3_carrier_book AS
SELECT carrier, AVG(CASE WHEN damaged THEN 1.0 ELSE 0 END) AS book_dmg_rate
FROM l1_order_economics GROUP BY carrier;

CREATE OR REPLACE TABLE l3_carrier_lift AS
WITH sc AS (
    SELECT sku, carrier, COUNT(*) AS shipped,
           AVG(CASE WHEN damaged THEN 1.0 ELSE 0 END) AS dmg_rate
    FROM l1_order_economics GROUP BY sku, carrier
)
SELECT sku, carrier AS top_damage_carrier, shipped AS top_carrier_shipped, dmg_rate AS top_carrier_dmg_rate,
       book_dmg_rate, dmg_rate / NULLIF(book_dmg_rate, 0) AS dmg_lift
FROM (
    SELECT sc.*, b.book_dmg_rate,
           ROW_NUMBER() OVER (PARTITION BY sc.sku ORDER BY sc.dmg_rate * sc.shipped DESC) AS rn
    FROM sc JOIN l3_carrier_book b USING (carrier)
) WHERE rn = 1;
