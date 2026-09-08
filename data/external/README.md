# data/external — generalization / validation databases

Downloaded + converted by `scripts/build_validation_dbs.py` (rerunnable; sources
are public production-style datasets — no bank/customer data). See
`manifest.json` for the authoritative inventory (rows per table, sizes, sources,
licenses).

| DB | Kind | Why it stresses the agent |
|----|------|---------------------------|
| chinook | SQLite | music-store sales, 11 tables, invoiceline joins |
| northwind | SQLite | 13-table ordering/HR/shipping web of FKs |
| sakila / sakila_sqlite | MySQL-dump / SQLite port | film-rental; same business domain twice, two different dump formats |
| world | MySQL dump | country/city/language demographics |
| menagerie | MySQL dump + LOAD-DATA TSVs | tiny (sanity scale) |
| classicmodels | MySQL dump (phpMyAdmin style) | orders/orderdetails/products/payments |
| f1db | ready-made SQLite | 31 tables of real F1 racing data, deep reference chains |
| openflights | CSV → SQLite | airlines/airports/routes, messy real-world codes |
| imdb | official TSVs → SQLite | 10.7M rows; movies↔ratings↔episodes joins at scale |
| tpch | duckdb dbgen (sf=0.3) → SQLite | TPC-H benchmark star schema, 2.6M rows |
| tpcds | duckdb dsdgen (sf=0.02) → SQLite | TPC-DS retail star schema, 24 tables |
| sas7bdat | real .sas7bdat files (pandas test corpus) | native SAS tables (airline, cars, productsales, …) read with pyreadstat |

All databases are SQLite under `sql/`; all `.sas7bdat` files are under
`sas/sas7bdat/`. Raw downloads cached in `downloads/` (git-ignored).

`db_debug_rl.pipelines` defines join pipelines on top of these DBs; defects are
planted into materialized pipeline outputs so ground truth remains correct by
construction (same mutation-testing contract as the synthetic `vendor/` set).
