"""Materialized reporting pipelines over the external production databases.

A pipeline is a small ordered program (staging steps -> final reporting table)
that mimics a SAS project's transformation layers. Each step is materialized as
a SQLite table. The final step builds the "reported" table whose values the
captain doubts. Every step reads source tables through the ``src`` attachment
(the downloaded database) and writes into an in-memory working DB.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    name: str
    layer: str
    description: str
    body: str          # SELECT body; materialized as CREATE TABLE name AS <body>


@dataclass(frozen=True)
class Pipeline:
    pipeline_id: str
    db: str
    domain: str
    final_table: str
    pk: str
    lineage: str
    steps: tuple[Step, ...]
    indexes: tuple[str, ...] = ()   # CREATE INDEX statements on staging join keys
    weight: str = "light"           # light = fast materialization; heavy = minuted

    @property
    def final_body(self) -> str:
        return self.steps[-1].body


PIPELINES: dict[str, Pipeline] = {}

_P = PIPELINES

_P["chinook_revenue"] = Pipeline(
    pipeline_id="chinook_revenue",
    db="chinook", domain="music retail", final_table="customer_revenue", pk="CustomerId",
    lineage=("L0 stg_lines/stg_invoices: staging of invoicelines and invoices.\n"
             "L1 aggregation: line_amount = UnitPrice * Quantity per invoiceline.\n"
             "L2 customer_revenue: per customer, n_invoices, revenue_calc from joined "
             "lines, revenue_ref from an independent recomputation."),
    steps=(
        Step("stg_lines", "L0", "staging invoicelines with line_amount",
             'SELECT il.InvoiceLineId, il.InvoiceId, il.TrackId, il.UnitPrice, '
             'il.Quantity, ROUND(il.UnitPrice * il.Quantity, 2) AS line_amount '
             'FROM src."InvoiceLine" il'),
        Step("stg_invoices", "L0", "staging invoices",
             'SELECT i.InvoiceId, i.CustomerId, i.Total FROM src."Invoice" i'),
        Step("customer_revenue", "L2", "per-customer revenue reconciliation",
             'SELECT c."CustomerId", c."FirstName", c."LastName", c."Country", '
             'COUNT(DISTINCT i."InvoiceId") AS n_invoices, '
             'ROUND(COALESCE(SUM(l.line_amount), 0), 2) AS revenue_calc, '
             'ROUND(COALESCE((SELECT SUM(Total) FROM src."Invoice" '
             'WHERE "CustomerId" = c."CustomerId"), 0), 2) AS revenue_ref '
             'FROM src."Customer" c '
             'LEFT JOIN stg_invoices i ON i."CustomerId" = c."CustomerId" '
             'LEFT JOIN stg_lines l ON l."InvoiceId" = i."InvoiceId" '
             'GROUP BY c."CustomerId", c."FirstName", c."LastName", c."Country"'),
    ),
    indexes=('CREATE INDEX ix_ci ON stg_invoices("CustomerId")',
             'CREATE INDEX ix_li ON stg_lines("InvoiceId")'),
)

_P["northwind_orders"] = Pipeline(
    pipeline_id="northwind_orders",
    db="northwind", domain="wholesale orders", final_table="order_value", pk="OrderID",
    lineage=("L0 stg_details/stg_products: staging of order details and product list prices.\n"
             "L1 order_value: per order, n_lines, gross_value, net_value (discounted), "
             "list_value from the joined product catalog, list_value_ref recomputed "
             "independently from source tables."),
    steps=(
        Step("stg_details", "L0", "staging order details with gross amount",
             'SELECT od."OrderID", od."ProductID", od.UnitPrice, od.Quantity, od.Discount, '
             'ROUND(od.UnitPrice * od.Quantity, 2) AS gross '
             'FROM src."Order Details" od'),
        Step("stg_products", "L0", "staging product list prices",
             'SELECT p."ProductID", p."ProductName", p."SupplierID", p."UnitPrice" AS list_price '
             'FROM src."Products" p'),
        Step("order_value", "L1", "per-order value reconciliation",
             'SELECT o."OrderID", o."CustomerID", o."OrderDate", o."ShipCountry", '
             'COUNT(*) AS n_lines, '
             'ROUND(SUM(d.gross), 2) AS gross_value, '
             'ROUND(COALESCE(SUM(d.gross * (1 - d.Discount)), 0), 2) AS net_value, '
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
    ),
    indexes=('CREATE INDEX ix_nd ON stg_details("OrderID")',
             'CREATE INDEX ix_np ON stg_products("ProductID")'),
    weight="heavy",
)

_P["sakila_films"] = Pipeline(
    pipeline_id="sakila_films",
    db="sakila_sqlite", domain="video rental", final_table="film_performance", pk="film_id",
    lineage=("L0 stg_inventory/stg_rentals/stg_payments: staging of copies, rentals and payments.\n"
             "L1 film_performance: per film, n_rentals, revenue_calc from the joined "
             "rental/payment chain, revenue_ref recomputed independently."),
    steps=(
        Step("stg_inventory", "L0", "staging of film copies",
             'SELECT inventory_id, film_id, store_id FROM src."inventory"'),
        Step("stg_rentals", "L0", "staging of rentals",
             'SELECT rental_id, inventory_id, customer_id, return_date IS NULL AS still_out '
             'FROM src."rental"'),
        Step("stg_payments", "L0", "staging of payments",
             'SELECT payment_id, customer_id, staff_id, rental_id, amount FROM src."payment"'),
        Step("film_performance", "L1", "per-film rental revenue reconciliation",
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
             'LEFT JOIN stg_payments p ON p.rental_id = r.rental_id '
             'GROUP BY f.film_id, f.title, f.rating, f.rental_rate'),
    ),
    indexes=('CREATE INDEX ix_sif ON stg_inventory(film_id)',
             'CREATE INDEX ix_sri ON stg_rentals(inventory_id)',
             'CREATE INDEX ix_spr ON stg_payments(rental_id)'),
)

_P["classicmodels_sales"] = Pipeline(
    pipeline_id="classicmodels_sales",
    db="classicmodels", domain="B2B sales", final_table="sales_by_customer", pk="customerNumber",
    lineage=("L0 stg_orders/stg_details/stg_payments: staging of orders, order lines and payments.\n"
             "L1 sales_by_customer: per customer, n_orders, revenue_calc from order lines, "
             "paid_calc from payments, revenue_ref recomputed independently."),
    steps=(
        Step("stg_orders", "L0", "staging of orders",
             'SELECT o."orderNumber", o."customerNumber", o."orderDate", o."status" '
             'FROM src."orders" o'),
        Step("stg_details", "L0", "staging of order lines with line_amount",
             'SELECT od."orderNumber", od."productCode", od."quantityOrdered", '
             'od."priceEach", ROUND(od."quantityOrdered" * od."priceEach", 2) AS line_amount '
             'FROM src."orderdetails" od'),
        Step("stg_payments", "L0", "staging of payments",
             'SELECT p."customerNumber", p."checkNumber", p."amount" FROM src."payments" p'),
        Step("sales_by_customer", "L1", "per-customer sales reconciliation",
             'SELECT cu."customerNumber", cu."customerName", cu."country", '
             'COUNT(DISTINCT o."orderNumber") AS n_orders, '
             'ROUND(COALESCE(SUM(d.line_amount), 0), 2) AS revenue_calc, '
             'ROUND(COALESCE((SELECT SUM(p2."amount") FROM stg_payments p2 '
             'WHERE p2."customerNumber" = cu."customerNumber"), 0), 2) AS paid_calc, '
             'ROUND(COALESCE((SELECT SUM(p3."amount") FROM stg_payments p3 '
             'WHERE p3."customerNumber" = cu."customerNumber"), 0), 2) AS paid_ref, '
             'ROUND(COALESCE((SELECT SUM(d2.line_amount) FROM stg_orders o2 '
             'JOIN stg_details d2 ON d2."orderNumber" = o2."orderNumber" '
             'WHERE o2."customerNumber" = cu."customerNumber"), 0), 2) AS revenue_ref '
             'FROM src."customers" cu '
             'LEFT JOIN stg_orders o ON o."customerNumber" = cu."customerNumber" '
             'LEFT JOIN stg_details d ON d."orderNumber" = o."orderNumber" '
             'GROUP BY cu."customerNumber", cu."customerName", cu."country"'),
    ),
    indexes=('CREATE INDEX ix_cso ON stg_orders("customerNumber")',
             'CREATE INDEX ix_csd ON stg_details("orderNumber")'),
)

_P["world_profile"] = Pipeline(
    pipeline_id="world_profile",
    db="world", domain="country demographics", final_table="country_profile", pk="Code",
    lineage=("L0 stg_cities/stg_languages: staging of cities and spoken languages.\n"
             "L1 country_profile: per country, n_cities, largest_city_pop, language_pct "
             "(sum of language percentages)."),
    steps=(
        Step("stg_cities", "L0", "staging of cities",
             'SELECT "ID", "Name", "CountryCode", "District", "Population" FROM src."city"'),
        Step("stg_languages", "L0", "staging of spoken languages",
             'SELECT "CountryCode", "Language", "IsOfficial", "Percentage" '
             'FROM src."countrylanguage"'),
        Step("country_profile", "L1", "per-country profile reconciliation",
             'SELECT c."Code", c."Name", c."Continent", c."Population", '
             'COUNT(DISTINCT ci."ID") AS n_cities, '
             '(SELECT MAX("Population") FROM stg_cities WHERE "CountryCode" = c."Code") '
             'AS largest_city_pop, '
             'ROUND(COALESCE((SELECT SUM("Percentage") FROM stg_languages '
             'WHERE "CountryCode" = c."Code"), 0), 1) AS language_pct, '
             'ROUND(COALESCE((SELECT SUM("Percentage") FROM stg_languages '
             'WHERE "CountryCode" = c."Code"), 0), 1) AS language_pct_ref '
             'FROM src."country" c '
             'LEFT JOIN stg_cities ci ON ci."CountryCode" = c."Code" '
             'GROUP BY c."Code", c."Name", c."Continent", c."Population"'),
    ),
)

_P["f1_seasons"] = Pipeline(
    pipeline_id="f1_seasons",
    db="f1db", domain="formula 1", final_table="driver_season", pk="row_key",
    lineage=("L0 stg_results/stg_races: staging of race-classified results and races.\n"
             "L1 driver_season: per driver per year, n_races, wins, podiums, avg_finish; "
             "wins_ref recomputed independently."),
    steps=(
        Step("stg_results", "L0", "staging of race results (type = race)",
             'SELECT rd.race_id, rd.driver_id, rd.constructor_id, rd.position_number '
             'FROM src."race_data" rd WHERE rd.type = \'RACE_RESULT\''),
        Step("stg_races", "L0", "staging of races",
             'SELECT r.id, r.year, r.round, r.grand_prix_id FROM src."race" r'),
        Step("driver_season", "L1", "per-driver season reconciliation",
             'SELECT r.year || \':\' || rd.driver_id AS row_key, r.year, rd.driver_id, '
             'd.full_name, COUNT(*) AS n_races, '
             'SUM(CASE WHEN rd.position_number = 1 THEN 1 ELSE 0 END) AS wins, '
             'SUM(CASE WHEN rd.position_number <= 3 THEN 1 ELSE 0 END) AS podiums, '
             'ROUND(AVG(rd.position_number), 2) AS avg_finish, '
             '(SELECT COUNT(*) FROM stg_results r2 JOIN stg_races r3 '
             'ON r3.id = r2.race_id WHERE r2.driver_id = rd.driver_id '
             'AND r3.year = r.year AND r2.position_number = 1) AS wins_ref '
             'FROM stg_results rd '
             'JOIN stg_races r ON r.id = rd.race_id '
             'JOIN src."driver" d ON d.id = rd.driver_id '
             'GROUP BY r.year, rd.driver_id, d.full_name'),
    ),
    indexes=('CREATE INDEX ix_frr ON stg_results(race_id, driver_id)',),
    weight="heavy",
)

_P["tpch_margin"] = Pipeline(
    pipeline_id="tpch_margin",
    db="tpch", domain="supply chain", final_table="part_margin", pk="row_key",
    lineage=("L0 stg_lines/stg_orders: staging of lineitems and OPEN orders.\n"
             "L1 part_margin: per part+supplier, total_qty, net_revenue_calc over open "
             "orders, net_revenue_ref recomputed independently."),
    steps=(
        Step("stg_lines", "L0", "staging of line items",
             'SELECT l.l_orderkey, l.l_partkey, l.l_suppkey, l.l_quantity, '
             'l.l_extendedprice, l.l_discount FROM src."lineitem" l'),
        Step("stg_orders", "L0", "staging of OPEN orders",
             'SELECT o.o_orderkey, o.o_custkey, o.o_orderstatus '
             'FROM src."orders" o WHERE o.o_orderstatus = \'O\''),
        Step("part_margin", "L1", "per part+supplier margin reconciliation",
             'SELECT l.l_partkey || \':\' || l.l_suppkey AS row_key, l.l_partkey, '
             'l.l_suppkey, SUM(l.l_quantity) AS total_qty, '
             'ROUND(SUM(l.l_extendedprice * (1 - l.l_discount)), 2) AS net_revenue_calc, '
             'ROUND(COALESCE((SELECT SUM(l2.l_extendedprice * (1 - l2.l_discount)) '
             'FROM stg_lines l2 JOIN stg_orders o2 ON o2.o_orderkey = l2.l_orderkey '
             'WHERE l2.l_partkey = l.l_partkey AND l2.l_suppkey = l.l_suppkey), 0), 2) '
             'AS net_revenue_ref '
             'FROM stg_lines l '
             'JOIN stg_orders o ON o.o_orderkey = l.l_orderkey '
             'GROUP BY l.l_partkey, l.l_suppkey'),
    ),
    indexes=('CREATE INDEX ix_tpo ON stg_orders(o_orderkey)',
             'CREATE INDEX ix_tpl ON stg_lines(l_partkey, l_suppkey)'),
    weight="heavy",
)

_P["tpcds_sales"] = Pipeline(
    pipeline_id="tpcds_sales",
    db="tpcds", domain="retail", final_table="item_sales", pk="i_item_sk",
    lineage=("L0 stg_sales/stg_dates/stg_items: staging of store sales, the date "
             "dimension and the item dimension.\n"
             "L1 item_sales: per item in year 2001, units_calc, sales_calc, sales_ref "
             "recomputed independently."),
    steps=(
        Step("stg_sales", "L0", "staging of store sales",
             'SELECT ss_sold_date_sk, ss_item_sk, ss_store_sk, ss_ticket_number, '
             'ss_quantity, ROUND(ss_quantity * ss_sales_price, 2) AS line_sales '
             'FROM src."store_sales"'),
        Step("stg_dates", "L0", "staging of the date dimension",
             'SELECT d_date_sk, d_year, d_moy FROM src."date_dim"'),
        Step("stg_items", "L0", "staging of the item dimension",
             'SELECT i_item_sk, i_item_id, i_category, i_class FROM src."item"'),
        Step("item_sales", "L1", "per-item year-2001 sales reconciliation",
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
             'WHERE d.d_year = 2001 '
             'GROUP BY i.i_item_sk, i.i_item_id, i.i_category'),
    ),
    indexes=('CREATE INDEX ix_tdd ON stg_dates(d_date_sk)',
             'CREATE INDEX ix_tds ON stg_sales(ss_item_sk)'),
    weight="heavy",
)

_P["openflights_routes"] = Pipeline(
    pipeline_id="openflights_routes",
    db="openflights", domain="air transport", final_table="route_stats", pk="airline_id",
    lineage=("L0 stg_routes/stg_airlines: staging of routes and airlines.\n"
             "L1 route_stats: per airline, n_routes_calc, direct_routes, n_destinations, "
             "n_routes_ref recomputed independently."),
    steps=(
        Step("stg_routes", "L0", "staging of routes",
             'SELECT r."airline", r."airline_id", r."src_airport", r."src_airport_id", '
             'r."dst_airport", r."dst_airport_id", r."codeshare", r."stops" '
             'FROM src."routes" r'),
        Step("stg_airlines", "L0", "staging of airlines",
             'SELECT a."airline_id", a."name", a."iata", a."icao", a."country", a."active" '
             'FROM src."airlines" a'),
        Step("route_stats", "L1", "per-airline route reconciliation",
             'SELECT a."airline_id", a."name", a."country", COUNT(*) AS n_routes_calc, '
             'SUM(CASE WHEN r."stops" = 0 THEN 1 ELSE 0 END) AS direct_routes, '
             'COUNT(DISTINCT r."dst_airport") AS n_destinations, '
             '(SELECT COUNT(*) FROM stg_routes r2 WHERE r2."airline_id" = a."airline_id") '
             'AS n_routes_ref '
             'FROM stg_routes r '
             'JOIN stg_airlines a ON a."airline_id" = r."airline_id" '
             'GROUP BY a."airline_id", a."name", a."country"'),
    ),
    indexes=('CREATE INDEX ix_orl ON stg_routes("airline_id")',
             'CREATE INDEX ix_ora ON stg_airlines("airline_id")'),
)

_P["imdb_titles"] = Pipeline(
    pipeline_id="imdb_titles",
    db="imdb", domain="movies", final_table="title_stats", pk="tconst",
    lineage=("L0 stg_titles/stg_ratings: staging of movies/series titles and ratings.\n"
             "L1 stg_episode_counts: episodes pre-aggregated per parent title.\n"
             "L2 title_stats: per title, average_rating, num_votes, n_episodes_calc vs "
             "n_episodes_ref recomputed independently."),
    steps=(
        Step("stg_titles", "L0", "staging of movie/series titles",
             'SELECT b."tconst", b."title_type", b."primary_title", b."start_year" '
             'FROM src."title_basics" b '
             'WHERE b."title_type" IN (\'movie\', \'tvSeries\')'),
        Step("stg_ratings", "L0", "staging of ratings",
             'SELECT r."tconst", r."average_rating", r."num_votes" '
             'FROM src."title_ratings" r'),
        Step("stg_episode_counts", "L1", "episodes pre-aggregated per parent title",
             'SELECT e."parent_tconst", COUNT(*) AS n_episodes '
             'FROM src."title_episode" e GROUP BY e."parent_tconst"'),
        Step("title_stats", "L2", "per-title reconciliation",
             'SELECT b."tconst", b."title_type", b."primary_title", b."start_year", '
             'r."average_rating", r."num_votes", '
             'COALESCE(e.n_episodes, 0) AS n_episodes_calc, '
             '(SELECT COUNT(*) FROM src."title_episode" te '
             'WHERE te."parent_tconst" = b."tconst") AS n_episodes_ref '
             'FROM stg_titles b '
             'LEFT JOIN stg_ratings r ON r."tconst" = b."tconst" '
             'LEFT JOIN stg_episode_counts e ON e."parent_tconst" = b."tconst"'),
    ),
    indexes=('CREATE INDEX ix_ibt ON stg_titles("tconst")',
             'CREATE INDEX ix_irt ON stg_ratings("tconst")',
             'CREATE INDEX ix_iec ON stg_episode_counts("parent_tconst")'),
    weight="heavy",
)


# ── zero-shot generalization probes (NOT part of the training set) ──────────
# Novel pipelines over unseen table paths; used to test whether the trained
# adapter applies the reconciliation contract to schemas it never saw.

_P["chinook_tracks"] = Pipeline(
    pipeline_id="chinook_tracks",
    db="chinook", domain="music catalog", final_table="track_stats", pk="TrackId",
    lineage=("L0 stg_tracks/stg_track_sales: staging of the track catalog (with "
             "genre and media type) and of the sold lines per track.\n"
             "L1 track_stats: per track, units_calc from the sold lines, "
             "revenue_calc from the sold lines, revenue_ref recomputed "
             "independently."),
    steps=(
        Step("stg_tracks", "L0", "staging of tracks with genre and media type",
             'SELECT t."TrackId", t."Name" AS track_name, g."Name" AS genre, '
             'm."Name" AS media_type FROM src."Track" t '
             'LEFT JOIN src."Genre" g ON g."GenreId" = t."GenreId" '
             'LEFT JOIN src."MediaType" m ON m."MediaTypeId" = t."MediaTypeId"'),
        Step("stg_track_sales", "L0", "staging of sold lines per track",
             'SELECT il."TrackId", il."InvoiceId", il."UnitPrice", il."Quantity", '
             'ROUND(il."UnitPrice" * il."Quantity", 2) AS line_amount '
             'FROM src."InvoiceLine" il'),
        Step("track_stats", "L1", "per-track sales reconciliation",
             'SELECT t."TrackId", t.track_name, t.genre, COUNT(*) AS n_lines, '
             'SUM(s."Quantity") AS units_calc, '
             'ROUND(COALESCE(SUM(s.line_amount), 0), 2) AS revenue_calc, '
             'ROUND(COALESCE((SELECT SUM(s2."UnitPrice" * s2."Quantity") '
             'FROM stg_track_sales s2 WHERE s2."TrackId" = t."TrackId"), 0), 2) '
             'AS revenue_ref '
             'FROM stg_tracks t '
             'LEFT JOIN stg_track_sales s ON s."TrackId" = t."TrackId" '
             'GROUP BY t."TrackId", t.track_name, t.genre'),
    ),
    indexes=('CREATE INDEX ix_tss ON stg_track_sales("TrackId")',),
)

_P["sakila_stores"] = Pipeline(
    pipeline_id="sakila_stores",
    db="sakila_sqlite", domain="video rental network", final_table="store_stats",
    pk="store_id",
    lineage=("L0 stg_customers: staging of customers enriched with address, "
             "city and country.\n"
             "L1 store_stats: per store, the manager name, n_customers_calc "
             "from the joined customer staging and n_customers_ref recomputed "
             "independently."),
    steps=(
        Step("stg_customers", "L0",
             "staging of customers with address, city and country",
             'SELECT c.customer_id, c.store_id, c.address_id, c.first_name, '
             'c.last_name, c.active, a.address, ct.city, co.country '
             'FROM src."customer" c '
             'JOIN src."address" a ON a.address_id = c.address_id '
             'JOIN src."city" ct ON ct.city_id = a.city_id '
             'JOIN src."country" co ON co.country_id = ct.country_id'),
        Step("store_stats", "L1", "per-store customer reconciliation",
             'SELECT s.store_id, st.first_name AS manager_first, '
             'st.last_name AS manager_last, '
             'COUNT(DISTINCT c.customer_id) AS n_customers_calc, '
             'SUM(CASE WHEN c.active = 1 THEN 1 ELSE 0 END) AS n_active_calc, '
             '(SELECT COUNT(*) FROM src."customer" c2 '
             'WHERE c2.store_id = s.store_id) AS n_customers_ref '
             'FROM src."store" s '
             'LEFT JOIN src."staff" st ON st.staff_id = s.manager_staff_id '
             'LEFT JOIN stg_customers c ON c.store_id = s.store_id '
             'GROUP BY s.store_id, st.first_name, st.last_name'),
    ),
    indexes=('CREATE INDEX ix_scs ON stg_customers(store_id)',),
)


def get_pipeline(pipeline_id: str) -> Pipeline:
    return PIPELINES[pipeline_id]
