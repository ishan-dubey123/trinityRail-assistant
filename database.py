import sqlite3
from datetime import datetime, timedelta
import random
import os

# DB_PATH = "trinity_rail.db"

# Force use /tmp on Vercel immediately
if os.getenv("VERCEL"):
    DB_PATH = "/tmp/trinity_rail.db"
else:
    DB_PATH = "trinity_rail.db"



def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn

def create_tables():
    """Creates the railcars and leases tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Main railcar table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS railcars (
            car_id INTEGER PRIMARY KEY,
            car_type TEXT NOT NULL,        -- 'tank', 'boxcar', 'flatcar'
            status TEXT NOT NULL,          -- 'idle', 'leased', 'maintenance'
            region TEXT NOT NULL,          -- 'north', 'south', 'east', 'west'
            last_inspection_date TEXT,     -- date as string YYYY-MM-DD
            commodity TEXT                 -- 'grain', 'chemicals', 'coal', 'none'
        )
    """)

    # Lease information table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leases (
            lease_id INTEGER PRIMARY KEY,
            car_id INTEGER,
            customer_name TEXT,
            start_date TEXT,
            end_date TEXT,
            monthly_rate REAL,
            FOREIGN KEY (car_id) REFERENCES railcars(car_id)
        )
    """)

    conn.commit()
    conn.close()

def seed_data():
    """Fills the database with realistic mock data."""
    conn = get_connection()
    cursor = conn.cursor()

    # Only seed if empty
    cursor.execute("SELECT COUNT(*) FROM railcars")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return

    car_types   = ["tank", "boxcar", "flatcar"]
    statuses    = ["idle", "leased", "maintenance"]
    regions     = ["north", "south", "east", "west"]
    commodities = ["grain", "chemicals", "coal", "none"]
    customers   = [
        "AgriCorp", "ChemTrans", "CoalFreight",
        "GrainShippers", "IndustrialLogistics", "PetroChem"
    ]

    today = datetime.today()

    # Create 100 mock railcars
    for car_id in range(1, 101):
        car_type  = random.choice(car_types)
        status    = random.choice(statuses)
        region    = random.choice(regions)
        commodity = random.choice(commodities)

        # Random inspection date — some overdue (>90 days ago), some recent
        days_ago = random.randint(10, 150)
        inspection_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        cursor.execute("""
            INSERT INTO railcars (car_id, car_type, status, region, last_inspection_date, commodity)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (car_id, car_type, status, region, inspection_date, commodity))

        # If leased, create a lease record
        if status == "leased":
            start = (today - timedelta(days=random.randint(30, 180))).strftime("%Y-%m-%d")
            end   = (today + timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")
            rate  = round(random.uniform(800, 3000), 2)

            cursor.execute("""
                INSERT INTO leases (car_id, customer_name, start_date, end_date, monthly_rate)
                VALUES (?, ?, ?, ?, ?)
            """, (car_id, random.choice(customers), start, end, rate))

    conn.commit()
    conn.close()
    print("✅ Database seeded with 100 mock railcars.")

def run_query(sql: str):
    """
    Safely runs a SELECT query and returns results as a list of dicts.
    Blocks any dangerous write operations.
    """
    # --- Safety Check: block dangerous SQL ---
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    sql_upper = sql.upper()

    for word in dangerous_keywords:
        if word in sql_upper:
            return {
                "success": False,
                "error": f"Blocked: query contains '{word}' which is not allowed.",
                "results": []
            }

    # --- Run the query safely ---
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        # Convert rows to plain list of dicts
        results = [dict(row) for row in rows]

        return {
            "success": True,
            "results": results,
            "row_count": len(results)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": []
        }

# --- Run setup when this file is executed directly ---
if __name__ == "__main__":
    create_tables()
    seed_data()
    print("✅ Database ready.")
