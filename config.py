import requests

def get_symbols():
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        response = requests.get(url)
        data = response.json()
        symbols = [s['symbol'].lower() for s in data['symbols'] if s['status'] == 'TRADING' and s['symbol'].lower().endswith('usdt')]
        return symbols
    except:
        # Fallback to popular USDT symbols
        return ["btcusdt", "ethusdt", "bnbusdt", "adausdt", "solusdt", "dotusdt", "maticusdt", "avaxusdt", "linkusdt", "uniusdt"]

SYMBOLS = get_symbols()
DB_PATH = "data/ticks.db"
BINANCE_WS = "wss://fstream.binance.com/ws"