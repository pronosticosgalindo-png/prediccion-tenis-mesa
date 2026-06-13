import sys
sys.stdout.reconfigure(encoding='utf-8')
from curl_cffi import requests
import sqlite3
import pandas as pd
import json
from datetime import datetime

print("Conectando a Sofascore con motor de Chrome (curl_cffi)...")

# ── FECHA AUTOMATICA ───────────────────────────────────────
FECHA_HOY = datetime.now().strftime('%Y-%m-%d')
print(f"Extrayendo partidos para: {FECHA_HOY}")

cookies = {
    '_cc_id': '665c530e0f6aa71a8a4cb0637145eaa7',
    '_ga': 'GA1.1.1956304625.1762201631',
    'AD_VERGE_SESSION_COOKIE_V1': '8f5072f4-2b49-4f5c-a190-6e99e705eb06',
    '_adv_uid': 'c97e1656-b739-4520-ad16-137ebb86720a',
    'hb_insticator_uid': '847eaa92-6071-4fbc-bd86-227c62f6fd6d',
    '_gcl_au': '1.1.835639633.1780018185',
    '_ga_HNQ9P9MGZR': 'deleted',
    '_lc2_fpi': 'a78faec1e09d--01kstzmsyb4d7w7byfqm46gx1b',
    '_lc2_fpi_meta': '%7B%22w%22%3A1780095805393%7D',
    '_adv_sid': '7799cd2d-4229-48c2-8496-4616a6a962cd',
    'ssp_test': 'control',
    '__gads': 'ID=86ba3bf017fab095:T=1762201627:RT=1781314847:S=ALNI_MYXKVhFZ-tm3js_y8AGbLqMYgLU_A',
    '__gpi': 'UID=000012a59574e3c7:T=1762201627:RT=1781314847:S=ALNI_MYIUpKlhX_9Lvn6I8c9rYA1mKByZQ',
    '__eoi': 'ID=9f47ba8481905a21:T=1780018205:RT=1781314847:S=AA-AfjZK16GEX47BwSM7nZyc3Mj9',
    'cto_bundle': '_-c0x19YUmFhbXozMUdhcE01QUt3Q3FCbU8lMkIyWXNnSDlFbFRjbXREOUVwa25lVjNyQmlqM1FyVVdEcDJ1YjdnZkc2MUdZRnZMR3BGbEhGYmFzbWxvWVZoMVFiM2ZIT255MFhSJTJGeXVVMjRaV1JwQ1lrblRvNnJZV0JPb3U5TTRsWHFhd0xLbFclMkZKbmJXMGsyTWQ0VUlqbFROU2clM0QlM0Q',
    'cto_bidid': '3LAkXV9pVWljNUhzZEZRQkdiclVtTUQ2bSUyQlBYZnlTb2p5WW8lMkJ5YlglMkZEbDdDbWpNNCUyRlRXcCUyRnJUVHB6c0ZSc0NJeE1uNzFkNkd5JTJGOEFsdzE3Y2E5amt4YzdhdlBrV09YSDFqSGYlMkY1UnJVVEMzV1VZJTNE',
    'FCCDCF': '%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%22398b09f5-8b5b-4d0d-8a97-49e5c52d93ce%5C%22%2C%5B1762201626%2C136000000%5D%5D%22%5D%5D%5D',
    'FCNEC': '%5B%5B%22AKsRol9BpLUSC47Z1hzd4T3DKtzbfCXIt78Ovnh3X6XVhfT0hLmBiX7zrYSp4daOfAh0f0Os3uUbJSrcwxbSCyAzvH2kZ5V1iVJ7syiAKAzJONqJa7DH8qwDKffLGnxX7W0CZV66mSVP6sq_aC0giVMvOEr97E3VKA%3D%3D%22%5D%5D',
    '_ga_HNQ9P9MGZR': 'GS2.1.s1781314860$o26$g1$t1781315052$j60$l0$h0',
}

headers = {
    'accept': '*/*',
    'accept-language': 'es-ES,es;q=0.9',
    'baggage': 'sentry-environment=production,sentry-public_key=d693747a6bb242d9bb9cf7069fb57988,sentry-trace_id=76d5559cc9daa2c9000af57eb70eccfd,sentry-org_id=18522,sentry-sample_rand=0.4194900655728808',
    'cache-control': 'max-age=0',
    'if-none-match': 'W/"4738ecccc9"',
    'priority': 'u=1, i',
    'referer': 'https://www.sofascore.com/es/table-tennis',
    'sec-ch-ua': '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'sentry-trace': '76d5559cc9daa2c9000af57eb70eccfd-b1d15ee05830f521',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36',
    'x-requested-with': 'd79e08',
}

# ── LIMPIEZA DE COLUMNAS COMPLEJAS ─────────────────────────
def limpiar_dataframe_para_sql(df):
    df = df.copy()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    return df

# ── ELIMINAR DUPLICADOS ANTES DE GUARDAR ───────────────────
def guardar_sin_duplicados(df_nuevos, conexion):
    try:
        df_existentes = pd.read_sql("SELECT id FROM matches", conexion)
        ids_existentes = set(df_existentes['id'].astype(str).tolist())
        df_filtrado = df_nuevos[
            ~df_nuevos['id'].astype(str).isin(ids_existentes)
        ]
    except Exception:
        df_filtrado = df_nuevos

    if len(df_filtrado) == 0:
        print("Sin partidos nuevos para guardar — todos ya existen en la base.")
        return 0

    df_filtrado.to_sql("matches", conexion, if_exists="append", index=False)
    return len(df_filtrado)

# ── EXTRACCION ─────────────────────────────────────────────
response = requests.get(
    f'https://www.sofascore.com/api/v1/unique-tournament/24047/scheduled-events/{FECHA_HOY}',
    cookies=cookies,
    headers=headers,
    impersonate="chrome120"
)

if response.status_code == 200:
    lista_partidos = response.json().get('events', [])

    if lista_partidos:
        df_nuevos = pd.json_normalize(lista_partidos)
        print(f"Partidos encontrados en Sofascore : {len(df_nuevos)}")

        df_nuevos = limpiar_dataframe_para_sql(df_nuevos)

        conexion = sqlite3.connect("base_ping_pong.sqlite")
        guardados = guardar_sin_duplicados(df_nuevos, conexion)
        conexion.close()

        print(f"Partidos nuevos guardados         : {guardados}")
        print(f"Partidos duplicados ignorados     : {len(df_nuevos) - guardados}")
        print("extractor.py ejecutado correctamente")
    else:
        print("Conexion exitosa, pero no hay partidos para esta fecha.")

else:
    print(f"Error al conectar. Codigo: {response.status_code}")
