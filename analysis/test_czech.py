import sys
sys.stdout.reconfigure(encoding='utf-8')
from curl_cffi import requests
import pandas as pd
import json
import sqlite3

print("Probando Czech Liga Pro...")

r = requests.get(
    'https://www.sofascore.com/api/v1/unique-tournament/19039/scheduled-events/2026-05-31',
    impersonate='chrome120',
    timeout=10
)

print(f"Status: {r.status_code}")

lista = r.json().get('events', [])
print(f"Partidos encontrados: {len(lista)}")

df = pd.json_normalize(lista)
print(f"Columnas: {len(df.columns)}")

for col in df.columns:
    if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
        df[col] = df[col].apply(
            lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
        )

conn = sqlite3.connect('base_ping_pong.sqlite')
try:
    df.to_sql('matches_czech_test', conn, if_exists='replace', index=False)
    print("OK - guardado correctamente en matches_czech_test")
except Exception as e:
    print(f"ERROR: {e}")
conn.close()