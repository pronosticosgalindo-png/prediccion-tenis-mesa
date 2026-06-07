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
    '_adv_sid': '4e728cd8-f741-4726-81f3-5f49f9ef9cc7',
    'ssp_test': 'control',
    '_li_dcdm_c': '.sofascore.com',
    '__gads': 'ID=86ba3bf017fab095:T=1762201627:RT=1780863181:S=ALNI_MYXKVhFZ-tm3js_y8AGbLqMYgLU_A',
    '__gpi': 'UID=000012a59574e3c7:T=1762201627:RT=1780863181:S=ALNI_MYIUpKlhX_9Lvn6I8c9rYA1mKByZQ',
    '__eoi': 'ID=9f47ba8481905a21:T=1780018205:RT=1780863181:S=AA-AfjZK16GEX47BwSM7nZyc3Mj9',
    'cto_bundle': 'YpoI319YUmFhbXozMUdhcE01QUt3Q3FCbU94TSUyRm5xTW53T0RQa3RZNiUyQmNiTFJraWJFTjglMkJCMTA2TnJqS3VZd1RNbFJpOFhINWFkV1UzQllZRGF6ZEdNMUl2Vk9iU09vZFYzZjVMdFZncmpKUllxYnQ0anVUalgzWGtNdnd6RVphNnY2Y2ZBalJ4RjF5R29uU1FiVGs4VkFwdkElM0QlM0Q',
    'cto_bidid': 'LyUu_l9pVWljNUhzZEZRQkdiclVtTUQ2bSUyQlBYZnlTb2p5WW8lMkJ5YlglMkZEbDdDbWpNNCUyRlRXcCUyRnJUVHB6c0ZSc0NJeE1uNzFkNkd5JTJGOEFsdzE3Y2E5amt4Yzdhb1ZNMEg2SnZSMTVYa3hseWR3Y3hhNCUzRA',
    'FCCDCF': '%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%22398b09f5-8b5b-4d0d-8a97-49e5c52d93ce%5C%22%2C%5B1762201626%2C136000000%5D%5D%22%5D%5D%5D',
    'FCNEC': '%5B%5B%22AKsRol-BvZAfhBkdcnpR7305MbcOO2QCbXRtvQmAq6bv0zf7oAotcS9TyslRuvdLbfhzCcD8R48MUz70oR7C2M-346sgdfWqjDI79SElu5JStnIyziurNszcBRMYJ5qG-0Gvlor8Oszh-0w5PlFpBdhdxe8m9ZWEOA%3D%3D%22%5D%5D',
    '_ga_HNQ9P9MGZR': 'GS2.1.s1780863179$o20$g1$t1780863406$j60$l0$h0',
}

headers = {
    'accept': '*/*',
    'accept-language': 'es-ES,es;q=0.9',
    'baggage': 'sentry-environment=production,sentry-public_key=d693747a6bb242d9bb9cf7069fb57988,sentry-trace_id=69e3f92db8e1e8f8b94cf2d09038d923,sentry-org_id=18522,sentry-sample_rand=0.10315664439291339',
    'cache-control': 'max-age=0',
    'if-none-match': 'W/"fc983423e7"',
    'priority': 'u=1, i',
    'referer': 'https://www.sofascore.com/es/table-tennis',
    'sec-ch-ua': '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'sentry-trace': '69e3f92db8e1e8f8b94cf2d09038d923-a0c8bb113c231bdd',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
    'x-requested-with': 'dcda2a',
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