import sqlite3
from config import DB_PATH
import pandas as pd




def get_connection():
 return sqlite3.connect(DB_PATH, check_same_thread=False)




def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ticks (
    ts TEXT,
    symbol TEXT,
    price REAL,
    size REAL
    )
    """)
    conn.commit()
    conn.close()




def insert_tick(ts, symbol, price, size):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
    "INSERT INTO ticks VALUES (?, ?, ?, ?)",
    (ts, symbol, price, size)
    )
    conn.commit()
    conn.close()




def fetch_ticks(symbol, minutes=60):
    conn = get_connection()
    query = f"""
    SELECT ts, price, size FROM ticks
    WHERE symbol = ?
    AND ts >= datetime('now', '-{minutes} minutes')
    ORDER BY ts
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()
    return df