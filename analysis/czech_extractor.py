import sys
sys.stdout.reconfigure(encoding='utf-8')
from curl_cffi import requests
import sqlite3
import pandas as pd
import json
from datetime import datetime

print("Extrayendo Czech Liga Pro del dia...")

FECHA_HOY = datetime.now().strftime('%Y-%m-%d')
print(f"Fecha: {FECHA_HOY}")

def limpiar_dataframe_para_sql(df):
    df = df.copy()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    return df

def guardar_sin_duplicados(df_nuevos, conexion):
    try:
        df_existentes  = pd.read_sql("SELECT id FROM matches_czech", conexion)
        ids_existentes = set(df_existentes['id'].astype(str).tolist())
        df_filtrado    = df_nuevos[
            ~df_nuevos['id'].astype(str).isin(ids_existentes)
        ]
    except Exception:
        df_filtrado = df_nuevos

    if len(df_filtrado) == 0:
        return 0

    # Guardar con replace para evitar conflictos de columnas
    try:
        df_filtrado.to_sql("matches_czech", conexion,
                           if_exists="append", index=False)
        return len(df_filtrado)
    except Exception:
        # Si falla por columnas diferentes, reconstruir tabla
        df_existente_full = pd.read_sql("SELECT * FROM matches_czech", conexion)
        df_combinado = pd.concat([df_existente_full, df_filtrado], ignore_index=True)
        df_combinado.drop_duplicates(subset='id', keep='last', inplace=True)
        df_combinado.to_sql("matches_czech", conexion,
                            if_exists="replace", index=False)
        return len(df_filtrado)

response = requests.get(
    f'https://www.sofascore.com/api/v1/unique-tournament/19039/scheduled-events/{FECHA_HOY}',
    impersonate="chrome120",
    timeout=10
)

if response.status_code == 200:
    lista_partidos = response.json().get('events', [])

    if lista_partidos:
        df_nuevos = pd.json_normalize(lista_partidos)
        df_nuevos['torneo_id']     = 19039
        df_nuevos['torneo_nombre'] = 'Czech Liga Pro'
        df_nuevos = limpiar_dataframe_para_sql(df_nuevos)

        conexion  = sqlite3.connect("base_ping_pong.sqlite")
        guardados = guardar_sin_duplicados(df_nuevos, conexion)
        conexion.close()

        print(f"Partidos encontrados  : {len(df_nuevos)}")
        print(f"Partidos nuevos       : {guardados}")
        print(f"Duplicados ignorados  : {len(df_nuevos) - guardados}")
    else:
        print("Sin partidos para hoy.")
else:
    print(f"Error al conectar. Codigo: {response.status_code}")

print("czech_extractor.py ejecutado correctamente")