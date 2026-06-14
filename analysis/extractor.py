import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
import json
from datetime import datetime
from playwright.sync_api import sync_playwright

print("Conectando a Sofascore con Playwright (Chrome real)...")

# ── CONFIGURACION ──────────────────────────────────────────
FECHA_HOY  = datetime.now().strftime('%Y-%m-%d')
TORNEO_ID  = '19039'
DB_PATH    = 'base_ping_pong.sqlite'

print(f"Extrayendo partidos para: {FECHA_HOY}")

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
        df_existentes  = pd.read_sql("SELECT id FROM matches", conexion)
        ids_existentes = set(df_existentes['id'].astype(str).tolist())
        df_filtrado    = df_nuevos[~df_nuevos['id'].astype(str).isin(ids_existentes)]
    except Exception:
        df_filtrado = df_nuevos

    if len(df_filtrado) == 0:
        print("Sin partidos nuevos — todos ya existen en la base.")
        return 0

    df_filtrado.to_sql("matches", conexion, if_exists="append", index=False)
    return len(df_filtrado)

# ── EXTRACCION CON PLAYWRIGHT ──────────────────────────────
def extraer_con_playwright():
    URL = f'https://www.sofascore.com/api/v1/unique-tournament/{TORNEO_ID}/scheduled-events/{FECHA_HOY}'

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36',
            locale='es-ES',
        )

        # Visitar primero la página principal para obtener cookies válidas
        page = context.new_page()
        print("Cargando Sofascore...")
        page.goto('https://www.sofascore.com/es/table-tennis', wait_until='networkidle', timeout=30000)
        page.wait_for_timeout(2000)

        # Ahora llamar la API con las cookies ya establecidas
        print(f"Llamando API: {URL}")
        response = page.request.get(URL)

        if response.status != 200:
            print(f"Error HTTP: {response.status}")
            browser.close()
            return None

        data = response.json()
        browser.close()
        return data.get('events', [])

# ── MAIN ───────────────────────────────────────────────────
try:
    lista_partidos = extraer_con_playwright()

    if lista_partidos is None:
        print("Error al conectar con Sofascore.")
        sys.exit(1)

    if not lista_partidos:
        print("Conexion exitosa, pero no hay partidos para esta fecha.")
        sys.exit(0)

    df_nuevos = pd.json_normalize(lista_partidos)
    print(f"Partidos encontrados en Sofascore : {len(df_nuevos)}")

    df_nuevos = limpiar_dataframe_para_sql(df_nuevos)

    conexion  = sqlite3.connect(DB_PATH)
    guardados = guardar_sin_duplicados(df_nuevos, conexion)
    conexion.close()

    print(f"Partidos nuevos guardados         : {guardados}")
    print(f"Partidos duplicados ignorados     : {len(df_nuevos) - guardados}")
    print("extractor.py ejecutado correctamente")

except Exception as e:
    print(f"Error general: {e}")
    sys.exit(1)
