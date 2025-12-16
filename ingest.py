import json
import threading
import websocket
from datetime import datetime
from db import insert_tick, init_db
from config import SYMBOLS, BINANCE_WS


init_db()

def on_message(ws, message):
    data = json.loads(message)
    ts = datetime.utcfromtimestamp(data['T'] / 1000).isoformat()
    insert_tick(ts, data['s'].lower(), float(data['p']), float(data['q']))

def start_ws(symbol):
    url = f"{BINANCE_WS}/{symbol}@trade"
    ws = websocket.WebSocketApp(url, on_message=on_message)
    ws.run_forever()

def start_ingestion():
    for sym in SYMBOLS:
        t = threading.Thread(target=start_ws, args=(sym,), daemon=True)
        t.start()

if __name__ == "__main__":
    start_ingestion()
    threading.Event().wait()