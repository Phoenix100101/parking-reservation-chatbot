import psycopg
from psycopg.rows import dict_row
from pathlib import Path

DATA_BASE = "postgres"
SCHEMA = "parking"


def get_connection() -> psycopg.Connection:
    database_url = f"postgresql://postgres:mysecretpassword@localhost:5432/{DATA_BASE}" # Replace username and password here
    return psycopg.connect(
        conninfo=database_url,
        connect_timeout=5,
        row_factory=dict_row
    )

def drop_all_tables():
    with get_connection() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {SCHEMA}.operating_hours")
        conn.execute(f"DROP TABLE IF EXISTS {SCHEMA}.reservations")
        conn.execute(f"DROP TABLE IF EXISTS {SCHEMA}.spaces")
        conn.execute(f"DROP TYPE IF EXISTS {SCHEMA}.reservation_status")
    print("All tables have been dropped")

def create_tables():
    with get_connection() as conn:
        conn.execute(Path("src/seed_data/sql_schema_script.sql").read_text())
    print("All tables, indexes and types have been created")

def seed_data():
    with get_connection() as conn:
        conn.execute(Path("src/seed_data/postgres_seed.sql").read_text())
    print("All data have been seed")

def validate_table_data():
    pass

if __name__ == '__main__':
    print("Drop tables if exist-----------------")
    drop_all_tables()
    print("Create tables------------------------")
    create_tables()
    print("Seed data----------------------------")
    seed_data()
    print("Check data---------------------------")
    validate_table_data()
