"""Join-focused defect catalog planted into the external-DB pipelines.

Ground truth is correct by construction: each defect replaces the final
materialization step of a pipeline with a mutated version (wrong join key,
missing dedup, double counting, filter drift, time boundary, formula drift,
inner-vs-left). The oracle diagnostic is a reconciliation diff between the
final table and ``zz_truth`` (the independently recomputed clean pipeline),
which by definition returns 0 rows on the clean DB and >=1 row on the trap DB.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from db_debug_rl.pipelines import PIPELINES, Pipeline

EPS = 0.011


def diff_oracle(pipeline: Pipeline, col: str, numeric: bool) -> str:
    f, t, pk = pipeline.final_table, "zz_truth", pipeline.pk
    neq = (f'f."{col}" IS NOT t."{col}"')
    if numeric:
        neq += (f' OR (f."{col}" IS NOT NULL AND t."{col}" IS NOT NULL AND '
                f'ABS(CAST(f."{col}" AS REAL) - CAST(t."{col}" AS REAL)) > {EPS})')
    return (
        f'SELECT CAST(f."{pk}" AS TEXT) AS flagged_pk FROM "{f}" f '
        f'LEFT JOIN {t} t ON t."{pk}" IS f."{pk}" '
        f'WHERE t."{pk}" IS NULL OR {neq} '
        f'UNION ALL '
        f'SELECT CAST(t."{pk}" AS TEXT) FROM {t} t '
        f'LEFT JOIN "{f}" f ON f."{pk}" IS t."{pk}" '
        f'WHERE f."{pk}" IS NULL')


@dataclass(frozen=True)
class Defect:
    defect_id: str
    pipeline_id: str
    kind: str
    dimension: str
    category: str
    severity: str
    description: str
    columns: tuple[str, ...]
    numeric: bool
    trap_sql: str          # mutated final-step SELECT body
    oracle_sql: str = field(init=False, repr=False, default="")
    gold_sql: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        pipe = PIPELINES[self.pipeline_id]
        object.__setattr__(
            self, "oracle_sql", diff_oracle(pipe, self.columns[0], self.numeric))
        object.__setattr__(self, "gold_sql", GOLD_SQL[self.defect_id])


# Short, agent-writable gold diagnostics: a row-level invariant on the
# reported table's corroborating columns where possible, otherwise a small
# upstream recomputation/anti-join. The selftest proves each one catches the
# planted rows on the trap DB and returns 0 rows on the clean DB.
GOLD_SQL: dict[str, str] = {
    "CH01": 'SELECT CAST("CustomerId" AS TEXT) AS flagged_pk FROM customer_revenue '
            'WHERE "revenue_calc" IS NOT "revenue_ref" '
            'OR ABS(CAST("revenue_calc" AS REAL) - CAST("revenue_ref" AS REAL)) > 0.011',
    "CH02": 'SELECT CAST("CustomerId" AS TEXT) AS flagged_pk FROM customer_revenue '
            'WHERE "revenue_calc" IS NOT "revenue_ref" '
            'OR ABS(CAST("revenue_calc" AS REAL) - CAST("revenue_ref" AS REAL)) > 0.011',
    "NW01": 'SELECT CAST(o."OrderID" AS TEXT) AS flagged_pk FROM order_value o '
            'WHERE o."list_value" IS NOT o."list_value_ref" '
            'OR ABS(CAST(o."list_value" AS REAL) - CAST(o."list_value_ref" AS REAL)) > 0.011',
    "NW02": 'SELECT CAST(o."OrderID" AS TEXT) AS flagged_pk FROM order_value o '
            'WHERE o."net_value" IS NOT o."net_value_ref" '
            'OR ABS(CAST(o."net_value" AS REAL) - CAST(o."net_value_ref" AS REAL)) > 0.011',
    "SK01": 'SELECT CAST("film_id" AS TEXT) AS flagged_pk FROM film_performance '
            'WHERE "revenue_calc" IS NOT "revenue_ref" '
            'OR ABS(CAST("revenue_calc" AS REAL) - CAST("revenue_ref" AS REAL)) > 0.011',
    "SK02": 'SELECT CAST("film_id" AS TEXT) AS flagged_pk FROM film_performance '
            'WHERE "revenue_calc" IS NOT "revenue_ref" '
            'OR ABS(CAST("revenue_calc" AS REAL) - CAST("revenue_ref" AS REAL)) > 0.011',
    "SK03": 'SELECT CAST(f.film_id AS TEXT) AS flagged_pk FROM src."film" f '
            'LEFT JOIN film_performance p ON p.film_id = f.film_id '
            'WHERE p.film_id IS NULL',
    "CM01": 'SELECT CAST("customerNumber" AS TEXT) AS flagged_pk '
            'FROM sales_by_customer WHERE "paid_calc" IS NOT "paid_ref" '
            'OR ABS(CAST("paid_calc" AS REAL) - CAST("paid_ref" AS REAL)) > 0.011',
    "CM02": 'SELECT CAST("customerNumber" AS TEXT) AS flagged_pk FROM sales_by_customer '
            'WHERE "revenue_calc" IS NOT "revenue_ref" '
            'OR ABS(CAST("revenue_calc" AS REAL) - CAST("revenue_ref" AS REAL)) > 0.011',
    "WO01": 'SELECT CAST("Code" AS TEXT) AS flagged_pk FROM country_profile '
            'WHERE "language_pct" IS NOT "language_pct_ref" '
            'OR ABS(CAST("language_pct" AS REAL) - '
            'CAST("language_pct_ref" AS REAL)) > 0.051',
    "WO02": 'SELECT CAST("Code" AS TEXT) AS flagged_pk FROM country_profile '
            'WHERE "language_pct" IS NOT "language_pct_ref" '
            'OR ABS(CAST("language_pct" AS REAL) - '
            'CAST("language_pct_ref" AS REAL)) > 0.051',
    "F101": 'SELECT CAST("row_key" AS TEXT) AS flagged_pk FROM driver_season '
            'WHERE "wins" IS NOT "wins_ref"',
    "F102": 'SELECT CAST("row_key" AS TEXT) AS flagged_pk FROM driver_season '
            'WHERE "wins" IS NOT "wins_ref"',
    "TP01": 'SELECT CAST("row_key" AS TEXT) AS flagged_pk FROM part_margin '
            'WHERE "net_revenue_calc" IS NOT "net_revenue_ref" '
            'OR ABS(CAST("net_revenue_calc" AS REAL) - '
            'CAST("net_revenue_ref" AS REAL)) > 0.011',
    "TP02": 'SELECT CAST("row_key" AS TEXT) AS flagged_pk FROM part_margin '
            'WHERE "net_revenue_calc" IS NOT "net_revenue_ref" '
            'OR ABS(CAST("net_revenue_calc" AS REAL) - '
            'CAST("net_revenue_ref" AS REAL)) > 0.011',
    "TD01": 'SELECT CAST("i_item_sk" AS TEXT) AS flagged_pk FROM item_sales '
            'WHERE "sales_calc" IS NOT "sales_ref" '
            'OR ABS(CAST("sales_calc" AS REAL) - CAST("sales_ref" AS REAL)) > 0.011',
    "TD02": 'SELECT CAST("i_item_sk" AS TEXT) AS flagged_pk FROM item_sales '
            'WHERE "sales_calc" IS NOT "sales_ref" '
            'OR ABS(CAST("sales_calc" AS REAL) - CAST("sales_ref" AS REAL)) > 0.011',
    "OF01": 'SELECT CAST("airline_id" AS TEXT) AS flagged_pk FROM route_stats '
            'WHERE "n_routes_calc" IS NOT "n_routes_ref"',
    "OF02": 'SELECT CAST("airline_id" AS TEXT) AS flagged_pk FROM route_stats '
            'WHERE "n_routes_calc" IS NOT "n_routes_ref"',
    "IM01": 'SELECT CAST(b."tconst" AS TEXT) AS flagged_pk FROM stg_titles b '
            'LEFT JOIN title_stats t ON t."tconst" = b."tconst" '
            'WHERE t."tconst" IS NULL',
    "IM02": 'SELECT CAST(b."tconst" AS TEXT) AS flagged_pk FROM stg_titles b '
            'LEFT JOIN title_stats t ON t."tconst" = b."tconst" '
            'WHERE t."tconst" IS NULL',

    # zero-shot probes (row-level + anti-join forms, same contract)
    "ZT01": 'SELECT CAST("TrackId" AS TEXT) AS flagged_pk FROM track_stats '
            'WHERE "revenue_calc" IS NOT "revenue_ref" '
            'OR ABS(CAST("revenue_calc" AS REAL) - CAST("revenue_ref" AS REAL)) > 0.011',
    "ZT02": 'SELECT CAST("TrackId" AS TEXT) AS flagged_pk FROM track_stats '
            'WHERE "revenue_calc" IS NOT "revenue_ref" '
            'OR ABS(CAST("revenue_calc" AS REAL) - CAST("revenue_ref" AS REAL)) > 0.011',
    "ZS01": 'SELECT CAST("store_id" AS TEXT) AS flagged_pk FROM store_stats '
            'WHERE "n_customers_calc" IS NOT "n_customers_ref"',
    "ZS02": 'SELECT CAST("store_id" AS TEXT) AS flagged_pk FROM store_stats '
            'WHERE "n_customers_calc" IS NOT "n_customers_ref"',
}


def _d(defect_id: str, pipeline_id: str, kind: str, dimension: str, category: str,
       severity: str, description: str, columns: tuple[str, ...], numeric: bool,
       trap_sql: str) -> Defect:
    return Defect(defect_id, pipeline_id, kind, dimension, category, severity,
                  description, columns, numeric, trap_sql)


DEFECTS: list[Defect] = [
    # ── chinook ──────────────────────────────────────────────────────────────
    _d("CH01", "chinook_revenue", "wrong_join_key", "consistency", "join", "high",
       "customer_revenue joins invoicelines on the wrong key (TrackId matched "
       "against InvoiceId), so revenue_calc no longer equals the recomputed "
       "revenue.", ("revenue_calc", "revenue_ref"), True,
       'SELECT c."CustomerId", c."FirstName", c."LastName", c."Country", '
       'COUNT(DISTINCT i."InvoiceId") AS n_invoices, '
       'ROUND(COALESCE(SUM(l.line_amount), 0), 2) AS revenue_calc, '
       'ROUND(COALESCE((SELECT SUM(Total) FROM src."Invoice" '
       'WHERE "CustomerId" = c."CustomerId"), 0), 2) AS revenue_ref '
       'FROM src."Customer" c '
       'LEFT JOIN stg_invoices i ON i."CustomerId" = c."CustomerId" '
       'LEFT JOIN stg_lines l ON l."TrackId" = i."InvoiceId" '
       'GROUP BY c."CustomerId", c."FirstName", c."LastName", c."Country"'),
    _d("CH02", "chinook_revenue", "missing_dedup", "consistency", "aggregation", "high",
       "customer_revenue joins PlaylistTrack into the revenue aggregation "
       "without deduplicating, so tracks present in playlists are counted once "
       "per playlist membership (double counting).",
       ("revenue_calc", "revenue_ref"), True,
       'SELECT c."CustomerId", c."FirstName", c."LastName", c."Country", '
       'COUNT(DISTINCT i."InvoiceId") AS n_invoices, '
       'ROUND(COALESCE(SUM(l.line_amount), 0), 2) AS revenue_calc, '
       'ROUND(COALESCE((SELECT SUM(Total) FROM src."Invoice" '
       'WHERE "CustomerId" = c."CustomerId"), 0), 2) AS revenue_ref '
       'FROM src."Customer" c '
       'LEFT JOIN stg_invoices i ON i."CustomerId" = c."CustomerId" '
       'LEFT JOIN stg_lines l ON l."InvoiceId" = i."InvoiceId" '
       'LEFT JOIN src."PlaylistTrack" pt ON pt."TrackId" = l."TrackId" '
       'GROUP BY c."CustomerId", c."FirstName", c."LastName", c."Country"'),

    # ── northwind ────────────────────────────────────────────────────────────
    _d("NW01", "northwind_orders", "wrong_join_key", "consistency", "join", "high",
       "order_value joins the product catalog on the wrong key (SupplierID "
       "matched against the line's ProductID), so list_value mixes products "
       "from the wrong supplier row.", ("list_value", "list_value_ref"), True,
       'SELECT o."OrderID", o."CustomerID", o."OrderDate", o."ShipCountry", '
       'COUNT(*) AS n_lines, '
       'ROUND(SUM(d.gross), 2) AS gross_value, '
       'ROUND(SUM(d.gross * (1 - d.Discount)), 2) AS net_value, '
       'ROUND(SUM(p.list_price * d.Quantity), 2) AS list_value, '
       'ROUND(COALESCE((SELECT SUM(p2.list_price * d2.Quantity) '
       'FROM stg_details d2 JOIN stg_products p2 ON p2."ProductID" = d2."ProductID" '
       'WHERE d2."OrderID" = o."OrderID"), 0), 2) AS list_value_ref '
       'FROM src."Orders" o '
       'JOIN stg_details d ON d."OrderID" = o."OrderID" '
       'JOIN stg_products p ON p."SupplierID" = d."ProductID" '
       'GROUP BY o."OrderID", o."CustomerID", o."OrderDate", o."ShipCountry"'),
    _d("NW02", "northwind_orders", "formula", "accuracy", "calculation", "high",
       "order_value computes net_value as the gross sum without applying the "
       "line discount, so net_value drifts from the discounted recomputation.",
       ("net_value", "net_value_ref"), True,
       'SELECT o."OrderID", o."CustomerID", o."OrderDate", o."ShipCountry", '
       'COUNT(*) AS n_lines, '
       'ROUND(SUM(d.gross), 2) AS gross_value, '
       'ROUND(SUM(d.gross), 2) AS net_value, '
       'ROUND(COALESCE((SELECT SUM(d3.gross * (1 - d3.Discount)) '
       'FROM stg_details d3 WHERE d3."OrderID" = o."OrderID"), 0), 2) '
       'AS net_value_ref, '
       'ROUND(SUM(p.list_price * d.Quantity), 2) AS list_value, '
       'ROUND(COALESCE((SELECT SUM(p2.list_price * d2.Quantity) '
       'FROM stg_details d2 JOIN stg_products p2 ON p2."ProductID" = d2."ProductID" '
       'WHERE d2."OrderID" = o."OrderID"), 0), 2) AS list_value_ref '
       'FROM src."Orders" o '
       'JOIN stg_details d ON d."OrderID" = o."OrderID" '
       'JOIN stg_products p ON p."ProductID" = d."ProductID" '
       'GROUP BY o."OrderID", o."CustomerID", o."OrderDate", o."ShipCountry"'),

    # ── sakila ───────────────────────────────────────────────────────────────
    _d("SK01", "sakila_films", "missing_dedup", "consistency", "aggregation", "high",
       "film_performance joins film_actor into the revenue aggregation, so "
       "payments are counted once per actor of the film (fan-out double "
       "counting).", ("revenue_calc", "revenue_ref"), True,
       'SELECT f.film_id, f.title, f.rating, f.rental_rate, '
       'COUNT(DISTINCT r.rental_id) AS n_rentals, '
       'ROUND(COALESCE(SUM(p.amount), 0), 2) AS revenue_calc, '
       'ROUND(COALESCE((SELECT SUM(p2.amount) FROM stg_payments p2 '
       'JOIN stg_rentals r2 ON r2.rental_id = p2.rental_id '
       'JOIN stg_inventory i2 ON i2.inventory_id = r2.inventory_id '
       'WHERE i2.film_id = f.film_id), 0), 2) AS revenue_ref '
       'FROM src."film" f '
       'LEFT JOIN src."film_actor" fa ON fa.film_id = f.film_id '
       'LEFT JOIN stg_inventory i ON i.film_id = f.film_id '
       'LEFT JOIN stg_rentals r ON r.inventory_id = i.inventory_id '
       'LEFT JOIN stg_payments p ON p.rental_id = r.rental_id '
       'GROUP BY f.film_id, f.title, f.rating, f.rental_rate'),
    _d("SK02", "sakila_films", "wrong_join_key", "consistency", "join", "high",
       "film_performance attaches payments on the wrong key (customer_id "
       "instead of rental_id), so every payment of the renting customer is "
       "counted for each rental.", ("revenue_calc", "revenue_ref"), True,
       'SELECT f.film_id, f.title, f.rating, f.rental_rate, '
       'COUNT(DISTINCT r.rental_id) AS n_rentals, '
       'ROUND(COALESCE(SUM(p.amount), 0), 2) AS revenue_calc, '
       'ROUND(COALESCE((SELECT SUM(p2.amount) FROM stg_payments p2 '
       'JOIN stg_rentals r2 ON r2.rental_id = p2.rental_id '
       'JOIN stg_inventory i2 ON i2.inventory_id = r2.inventory_id '
       'WHERE i2.film_id = f.film_id), 0), 2) AS revenue_ref '
       'FROM src."film" f '
       'LEFT JOIN stg_inventory i ON i.film_id = f.film_id '
       'LEFT JOIN stg_rentals r ON r.inventory_id = i.inventory_id '
       'LEFT JOIN stg_payments p ON p.customer_id = r.customer_id '
       'GROUP BY f.film_id, f.title, f.rating, f.rental_rate'),
    _d("SK03", "sakila_films", "inner_vs_left", "completeness", "join", "medium",
       "film_performance uses inner joins, so films without any rental "
       "disappear from the reported table.", ("revenue_calc", "revenue_ref"), True,
       'SELECT f.film_id, f.title, f.rating, f.rental_rate, '
       'COUNT(DISTINCT r.rental_id) AS n_rentals, '
       'ROUND(COALESCE(SUM(p.amount), 0), 2) AS revenue_calc, '
       'ROUND(COALESCE((SELECT SUM(p2.amount) FROM stg_payments p2 '
       'JOIN stg_rentals r2 ON r2.rental_id = p2.rental_id '
       'JOIN stg_inventory i2 ON i2.inventory_id = r2.inventory_id '
       'WHERE i2.film_id = f.film_id), 0), 2) AS revenue_ref '
       'FROM src."film" f '
       'JOIN stg_inventory i ON i.film_id = f.film_id '
       'JOIN stg_rentals r ON r.inventory_id = i.inventory_id '
       'JOIN stg_payments p ON p.rental_id = r.rental_id '
       'GROUP BY f.film_id, f.title, f.rating, f.rental_rate'),

    # ── classicmodels ────────────────────────────────────────────────────────
    _d("CM01", "classicmodels_sales", "double_count_fanout", "consistency", "aggregation",
       "high",
       "sales_by_customer mixes order lines and payments (two independent 1:N "
       "relationships) in the same grouped join, so paid_calc is multiplied by "
       "the number of order lines per customer.",
       ("paid_calc", "paid_ref"), True,
       'SELECT cu."customerNumber", cu."customerName", cu."country", '
       'COUNT(DISTINCT o."orderNumber") AS n_orders, '
       'ROUND(COALESCE(SUM(d.line_amount), 0), 2) AS revenue_calc, '
       'ROUND(SUM(pay."amount"), 2) AS paid_calc, '
       'ROUND(COALESCE((SELECT SUM(p3."amount") FROM stg_payments p3 '
       'WHERE p3."customerNumber" = cu."customerNumber"), 0), 2) AS paid_ref, '
       'ROUND(COALESCE((SELECT SUM(d2.line_amount) FROM stg_orders o2 '
       'JOIN stg_details d2 ON d2."orderNumber" = o2."orderNumber" '
       'WHERE o2."customerNumber" = cu."customerNumber"), 0), 2) AS revenue_ref '
       'FROM src."customers" cu '
       'LEFT JOIN stg_orders o ON o."customerNumber" = cu."customerNumber" '
       'LEFT JOIN stg_details d ON d."orderNumber" = o."orderNumber" '
       'LEFT JOIN stg_payments pay ON pay."customerNumber" = cu."customerNumber" '
       'GROUP BY cu."customerNumber", cu."customerName", cu."country"'),
    _d("CM02", "classicmodels_sales", "filter_drift", "accuracy", "filter", "medium",
       "sales_by_customer only aggregates orders with status 'Shipped', so "
       "revenue for orders in other statuses is lost versus the reference "
       "recomputation.", ("revenue_calc", "revenue_ref"), True,
       'SELECT cu."customerNumber", cu."customerName", cu."country", '
       'COUNT(DISTINCT o."orderNumber") AS n_orders, '
       'ROUND(COALESCE(SUM(d.line_amount), 0), 2) AS revenue_calc, '
       'ROUND(COALESCE((SELECT SUM(p3."amount") FROM stg_payments p3 '
       'WHERE p3."customerNumber" = cu."customerNumber"), 0), 2) AS paid_ref, '
       'ROUND(COALESCE((SELECT SUM(d2.line_amount) FROM stg_orders o2 '
       'JOIN stg_details d2 ON d2."orderNumber" = o2."orderNumber" '
       'WHERE o2."customerNumber" = cu."customerNumber"), 0), 2) AS revenue_ref '
       'FROM src."customers" cu '
       'LEFT JOIN stg_orders o ON o."customerNumber" = cu."customerNumber" '
       'AND o."status" = \'Shipped\' '
       'LEFT JOIN stg_details d ON d."orderNumber" = o."orderNumber" '
       'GROUP BY cu."customerNumber", cu."customerName", cu."country"'),

    # ── world ────────────────────────────────────────────────────────────────
    _d("WO01", "world_profile", "double_count_fanout", "consistency", "aggregation",
       "high",
       "country_profile joins both cities and languages in the same grouped "
       "query and sums language percentages over the cross product, inflating "
       "language_pct by the number of cities.", ("language_pct",), True,
       'SELECT c."Code", c."Name", c."Continent", c."Population", '
       'COUNT(DISTINCT ci."ID") AS n_cities, '
       '(SELECT MAX("Population") FROM stg_cities WHERE "CountryCode" = c."Code") '
       'AS largest_city_pop, '
       'ROUND(COALESCE(SUM(cl."Percentage"), 0), 1) AS language_pct, '
       'ROUND(COALESCE((SELECT SUM("Percentage") FROM stg_languages '
       'WHERE "CountryCode" = c."Code"), 0), 1) AS language_pct_ref '
       'FROM src."country" c '
       'LEFT JOIN stg_cities ci ON ci."CountryCode" = c."Code" '
       'LEFT JOIN stg_languages cl ON cl."CountryCode" = c."Code" '
       'GROUP BY c."Code", c."Name", c."Continent", c."Population"'),
    _d("WO02", "world_profile", "wrong_join_key", "consistency", "join", "high",
       "country_profile matches languages on the wrong key (country Region "
       "compared against CountryCode), so language_pct collapses for most "
       "countries.", ("language_pct",), True,
       'SELECT c."Code", c."Name", c."Continent", c."Population", '
       'COUNT(DISTINCT ci."ID") AS n_cities, '
       '(SELECT MAX("Population") FROM stg_cities WHERE "CountryCode" = c."Code") '
       'AS largest_city_pop, '
       'ROUND(COALESCE((SELECT SUM("Percentage") FROM stg_languages '
       'WHERE "CountryCode" = c."Region"), 0), 1) AS language_pct, '
       'ROUND(COALESCE((SELECT SUM("Percentage") FROM stg_languages '
       'WHERE "CountryCode" = c."Code"), 0), 1) AS language_pct_ref '
       'FROM src."country" c '
       'LEFT JOIN stg_cities ci ON ci."CountryCode" = c."Code" '
       'GROUP BY c."Code", c."Name", c."Continent", c."Population"'),

    # ── f1db ─────────────────────────────────────────────────────────────────
    _d("F101", "f1_seasons", "wrong_join_key", "consistency", "join", "high",
       "driver_season joins races on the wrong key (round matched against race "
       "id), so results are attributed to the wrong years and wins drift from "
       "the recomputation.", ("wins", "n_races"), True,
       'SELECT r.year || \':\' || rd.driver_id AS row_key, r.year, rd.driver_id, '
       'd.full_name, COUNT(*) AS n_races, '
       'SUM(CASE WHEN rd.position_number = 1 THEN 1 ELSE 0 END) AS wins, '
       'SUM(CASE WHEN rd.position_number <= 3 THEN 1 ELSE 0 END) AS podiums, '
       'ROUND(AVG(rd.position_number), 2) AS avg_finish, '
       '(SELECT COUNT(*) FROM stg_results r2 JOIN stg_races r3 '
       'ON r3.id = r2.race_id WHERE r2.driver_id = rd.driver_id '
       'AND r3.year = r.year AND r2.position_number = 1) AS wins_ref '
       'FROM stg_results rd '
       'JOIN stg_races r ON r.round = rd.race_id '
       'JOIN src."driver" d ON d.id = rd.driver_id '
       'GROUP BY r.year, rd.driver_id, d.full_name'),
    _d("F102", "f1_seasons", "formula", "accuracy", "calculation", "high",
       "driver_season counts wins with position_number >= 1 instead of = 1, so "
       "every classified race is counted as a win.",
       ("wins", "wins_ref"), True,
       'SELECT r.year || \':\' || rd.driver_id AS row_key, r.year, rd.driver_id, '
       'd.full_name, COUNT(*) AS n_races, '
       'SUM(CASE WHEN rd.position_number >= 1 THEN 1 ELSE 0 END) AS wins, '
       'SUM(CASE WHEN rd.position_number <= 3 THEN 1 ELSE 0 END) AS podiums, '
       'ROUND(AVG(rd.position_number), 2) AS avg_finish, '
       '(SELECT COUNT(*) FROM stg_results r2 JOIN stg_races r3 '
       'ON r3.id = r2.race_id WHERE r2.driver_id = rd.driver_id '
       'AND r3.year = r.year AND r2.position_number = 1) AS wins_ref '
       'FROM stg_results rd '
       'JOIN stg_races r ON r.id = rd.race_id '
       'JOIN src."driver" d ON d.id = rd.driver_id '
       'GROUP BY r.year, rd.driver_id, d.full_name'),

    # ── tpch ─────────────────────────────────────────────────────────────────
    _d("TP01", "tpch_margin", "formula", "accuracy", "calculation", "high",
       "part_margin computes net_revenue as the raw extended price without "
       "applying the line discount.", ("net_revenue_calc", "net_revenue_ref"), True,
       'SELECT l.l_partkey || \':\' || l.l_suppkey AS row_key, l.l_partkey, '
       'l.l_suppkey, SUM(l.l_quantity) AS total_qty, '
       'ROUND(SUM(l.l_extendedprice), 2) AS net_revenue_calc, '
       'ROUND(COALESCE((SELECT SUM(l2.l_extendedprice * (1 - l2.l_discount)) '
       'FROM stg_lines l2 JOIN stg_orders o2 ON o2.o_orderkey = l2.l_orderkey '
       'WHERE l2.l_partkey = l.l_partkey AND l2.l_suppkey = l.l_suppkey), 0), 2) '
       'AS net_revenue_ref '
       'FROM stg_lines l '
       'JOIN stg_orders o ON o.o_orderkey = l.l_orderkey '
       'GROUP BY l.l_partkey, l.l_suppkey'),
    _d("TP02", "tpch_margin", "wrong_join_key", "consistency", "join", "high",
       "part_margin joins orders on the wrong key (customer key used as order "
       "key), so revenue includes unrelated open orders of the customer.",
       ("net_revenue_calc", "net_revenue_ref"), True,
       'SELECT l.l_partkey || \':\' || l.l_suppkey AS row_key, l.l_partkey, '
       'l.l_suppkey, SUM(l.l_quantity) AS total_qty, '
       'ROUND(SUM(l.l_extendedprice * (1 - l.l_discount)), 2) AS net_revenue_calc, '
       'ROUND(COALESCE((SELECT SUM(l2.l_extendedprice * (1 - l2.l_discount)) '
       'FROM stg_lines l2 JOIN stg_orders o2 ON o2.o_orderkey = l2.l_orderkey '
       'WHERE l2.l_partkey = l.l_partkey AND l2.l_suppkey = l.l_suppkey), 0), 2) '
       'AS net_revenue_ref '
       'FROM stg_lines l '
       'JOIN stg_orders o ON o.o_custkey = l.l_orderkey '
       'GROUP BY l.l_partkey, l.l_suppkey'),

    # ── tpcds ────────────────────────────────────────────────────────────────
    _d("TD01", "tpcds_sales", "time_boundary", "timeliness", "filter", "medium",
       "item_sales filters the date dimension on January (d_moy = 1) instead of "
       "the full year 2001, so sales outside January are lost.",
       ("sales_calc", "sales_ref"), True,
       'SELECT i.i_item_sk, i.i_item_id, i.i_category, '
       'COUNT(DISTINCT s.ss_ticket_number) AS n_tickets, '
       'SUM(s.ss_quantity) AS units_calc, '
       'ROUND(SUM(s.line_sales), 2) AS sales_calc, '
       'ROUND(COALESCE((SELECT SUM(s2.line_sales) FROM stg_sales s2 '
       'JOIN stg_dates d2 ON d2.d_date_sk = s2.ss_sold_date_sk '
       'WHERE s2.ss_item_sk = i.i_item_sk AND d2.d_year = 2001), 0), 2) '
       'AS sales_ref '
       'FROM stg_sales s '
       'JOIN stg_items i ON i.i_item_sk = s.ss_item_sk '
       'JOIN stg_dates d ON d.d_date_sk = s.ss_sold_date_sk '
       'WHERE d.d_moy = 1 '
       'GROUP BY i.i_item_sk, i.i_item_id, i.i_category'),
    _d("TD02", "tpcds_sales", "wrong_join_key", "consistency", "join", "high",
       "item_sales joins the item dimension on the wrong key (store key used "
       "as item key), so sales are attributed to the wrong items.",
       ("sales_calc", "sales_ref"), True,
       'SELECT i.i_item_sk, i.i_item_id, i.i_category, '
       'COUNT(DISTINCT s.ss_ticket_number) AS n_tickets, '
       'SUM(s.ss_quantity) AS units_calc, '
       'ROUND(SUM(s.line_sales), 2) AS sales_calc, '
       'ROUND(COALESCE((SELECT SUM(s2.line_sales) FROM stg_sales s2 '
       'JOIN stg_dates d2 ON d2.d_date_sk = s2.ss_sold_date_sk '
       'WHERE s2.ss_item_sk = i.i_item_sk AND d2.d_year = 2001), 0), 2) '
       'AS sales_ref '
       'FROM stg_sales s '
       'JOIN stg_items i ON i.i_item_sk = s.ss_store_sk '
       'JOIN stg_dates d ON d.d_date_sk = s.ss_sold_date_sk '
       'WHERE d.d_year = 2001 '
       'GROUP BY i.i_item_sk, i.i_item_id, i.i_category'),

    # ── openflights ──────────────────────────────────────────────────────────
    _d("OF01", "openflights_routes", "wrong_join_key", "consistency", "join", "high",
       "route_stats matches airlines by IATA code instead of internal "
       "airline_id, so routes of codeless airlines drop out and duplicate "
       "codes double count.", ("n_routes_calc", "n_routes_ref"), True,
       'SELECT a."airline_id", a."name", a."country", COUNT(*) AS n_routes_calc, '
       'SUM(CASE WHEN r."stops" = 0 THEN 1 ELSE 0 END) AS direct_routes, '
       'COUNT(DISTINCT r."dst_airport") AS n_destinations, '
       '(SELECT COUNT(*) FROM stg_routes r2 WHERE r2."airline_id" = a."airline_id") '
       'AS n_routes_ref '
       'FROM stg_routes r '
       'JOIN stg_airlines a ON a."iata" = r."airline" '
       'GROUP BY a."airline_id", a."name", a."country"'),
    _d("OF02", "openflights_routes", "filter_drift", "accuracy", "filter", "medium",
       "route_stats counts only codeshare routes, so non-codeshare routes are "
       "lost versus the reference recomputation.",
       ("n_routes_calc", "n_routes_ref"), True,
       'SELECT a."airline_id", a."name", a."country", COUNT(*) AS n_routes_calc, '
       'SUM(CASE WHEN r."stops" = 0 THEN 1 ELSE 0 END) AS direct_routes, '
       'COUNT(DISTINCT r."dst_airport") AS n_destinations, '
       '(SELECT COUNT(*) FROM stg_routes r2 WHERE r2."airline_id" = a."airline_id") '
       'AS n_routes_ref '
       'FROM stg_routes r '
       'JOIN stg_airlines a ON a."airline_id" = r."airline_id" '
       'WHERE r."codeshare" = \'Y\' '
       'GROUP BY a."airline_id", a."name", a."country"'),

    # ── imdb ─────────────────────────────────────────────────────────────────
    _d("IM01", "imdb_titles", "filter_drift", "completeness", "filter", "medium",
       "title_stats keeps only movies, so every series row is missing from the "
       "reported table.", ("n_episodes_calc", "n_episodes_ref"), True,
       'SELECT b."tconst", b."title_type", b."primary_title", b."start_year", '
       'r."average_rating", r."num_votes", '
       'COALESCE(e.n_episodes, 0) AS n_episodes_calc, '
       '(SELECT COUNT(*) FROM src."title_episode" te '
       'WHERE te."parent_tconst" = b."tconst") AS n_episodes_ref '
       'FROM stg_titles b '
       'LEFT JOIN stg_ratings r ON r."tconst" = b."tconst" '
       'LEFT JOIN stg_episode_counts e ON e."parent_tconst" = b."tconst" '
       'WHERE b."title_type" = \'movie\''),
    _d("IM02", "imdb_titles", "inner_vs_left", "completeness", "join", "medium",
       "title_stats inner-joins ratings, so titles without a rating disappear "
       "from the reported table.", ("average_rating", "n_episodes_calc"), True,
       'SELECT b."tconst", b."title_type", b."primary_title", b."start_year", '
       'r."average_rating", r."num_votes", '
       'COALESCE(e.n_episodes, 0) AS n_episodes_calc, '
       '(SELECT COUNT(*) FROM src."title_episode" te '
       'WHERE te."parent_tconst" = b."tconst") AS n_episodes_ref '
       'FROM stg_titles b '
       'JOIN stg_ratings r ON r."tconst" = b."tconst" '
       'LEFT JOIN stg_episode_counts e ON e."parent_tconst" = b."tconst"'),
    # ── zero-shot generalization probes (never trained on) ───────────────────
    _d("ZT01", "chinook_tracks", "wrong_join_key", "consistency", "join", "high",
       "track_stats joins the sold lines on the wrong key (invoice id matched "
       "against track id), so sales are attributed to the wrong tracks.",
       ("units_calc", "revenue_calc"), True,
       'SELECT t."TrackId", t.track_name, t.genre, COUNT(*) AS n_lines, '
       'SUM(s."Quantity") AS units_calc, '
       'ROUND(COALESCE(SUM(s.line_amount), 0), 2) AS revenue_calc, '
       'ROUND(COALESCE((SELECT SUM(s2."UnitPrice" * s2."Quantity") '
       'FROM stg_track_sales s2 WHERE s2."TrackId" = t."TrackId"), 0), 2) '
       'AS revenue_ref '
       'FROM stg_tracks t '
       'LEFT JOIN stg_track_sales s ON s."InvoiceId" = t."TrackId" '
       'GROUP BY t."TrackId", t.track_name, t.genre'),
    _d("ZT02", "chinook_tracks", "missing_dedup", "consistency", "aggregation", "high",
       "track_stats joins PlaylistTrack into the sales aggregation without "
       "deduplicating, so sales are counted once per playlist membership.",
       ("revenue_calc", "revenue_ref"), True,
       'SELECT t."TrackId", t.track_name, t.genre, COUNT(*) AS n_lines, '
       'SUM(s."Quantity") AS units_calc, '
       'ROUND(COALESCE(SUM(s.line_amount), 0), 2) AS revenue_calc, '
       'ROUND(COALESCE((SELECT SUM(s2."UnitPrice" * s2."Quantity") '
       'FROM stg_track_sales s2 WHERE s2."TrackId" = t."TrackId"), 0), 2) '
       'AS revenue_ref '
       'FROM stg_tracks t '
       'LEFT JOIN stg_track_sales s ON s."TrackId" = t."TrackId" '
       'LEFT JOIN src."PlaylistTrack" pt ON pt."TrackId" = t."TrackId" '
       'GROUP BY t."TrackId", t.track_name, t.genre'),
    _d("ZS01", "sakila_stores", "wrong_join_key", "consistency", "join", "high",
       "store_stats attaches customers on the wrong key (address_id matched "
       "against store_id), so customer counts collapse and mix across "
       "stores.", ("n_customers_calc", "n_customers_ref"), True,
       'SELECT s.store_id, st.first_name AS manager_first, '
       'st.last_name AS manager_last, '
       'COUNT(DISTINCT c.customer_id) AS n_customers_calc, '
       'SUM(CASE WHEN c.active = 1 THEN 1 ELSE 0 END) AS n_active_calc, '
       '(SELECT COUNT(*) FROM src."customer" c2 '
       'WHERE c2.store_id = s.store_id) AS n_customers_ref '
       'FROM src."store" s '
       'LEFT JOIN src."staff" st ON st.staff_id = s.manager_staff_id '
       'LEFT JOIN stg_customers c ON c.address_id = s.store_id '
       'GROUP BY s.store_id, st.first_name, st.last_name'),
    _d("ZS02", "sakila_stores", "filter_drift", "completeness", "filter", "medium",
       "store_stats only counts inactive customers, so all active customers "
       "are lost from the reported counts.",
       ("n_customers_calc", "n_customers_ref"), True,
       'SELECT s.store_id, st.first_name AS manager_first, '
       'st.last_name AS manager_last, '
       'COUNT(DISTINCT c.customer_id) AS n_customers_calc, '
       'SUM(CASE WHEN c.active = 1 THEN 1 ELSE 0 END) AS n_active_calc, '
       '(SELECT COUNT(*) FROM src."customer" c2 '
       'WHERE c2.store_id = s.store_id) AS n_customers_ref '
       'FROM src."store" s '
       'LEFT JOIN src."staff" st ON st.staff_id = s.manager_staff_id '
       'LEFT JOIN stg_customers c ON c.store_id = s.store_id '
       'AND c.active = 0 '
       'GROUP BY s.store_id, st.first_name, st.last_name'),
]

DEFECTS_BY_ID: dict[str, Defect] = {d.defect_id: d for d in DEFECTS}


def defects_for_pipeline(pipeline_id: str) -> list[Defect]:
    return [d for d in DEFECTS if d.pipeline_id == pipeline_id]
