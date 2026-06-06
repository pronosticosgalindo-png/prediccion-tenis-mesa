import sys
sys.stdout.reconfigure(encoding='utf-8')
from curl_cffi import requests
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import time

print("Extrayendo historial Czech Liga Pro...")
print("="*55)

FECHA_INICIO = datetime(2026, 3, 2)
FECHA_FIN    = datetime.now()

total_dias = (FECHA_FIN - FECHA_INICIO).days + 1
print(f"Desde  : {FECHA_INICIO.strftime('%Y-%m-%d')}")
print(f"Hasta  : {FECHA_FIN.strftime('%Y-%m-%d')}")
print(f"Dias   : {total_dias}")
print(f"Torneo : [19039] Czech Liga Pro")
print("="*55)

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
        df_existentes = pd.read_sql("SELECT id FROM matches_czech", conexion)
        ids_existentes = set(df_existentes['id'].astype(str).tolist())
        df_filtrado = df_nuevos[
            ~df_nuevos['id'].astype(str).isin(ids_existentes)
        ]
    except Exception:
        df_filtrado = df_nuevos

    if len(df_filtrado) == 0:
        return 0

    df_filtrado.to_sql("matches_czech", conexion, if_exists="append", index=False)
    return len(df_filtrado)

conexion         = sqlite3.connect("base_ping_pong.sqlite")
total_guardados  = 0
total_duplicados = 0
dias_con_datos   = 0
dias_sin_datos   = 0
errores          = 0
fecha_actual     = FECHA_INICIO

while fecha_actual <= FECHA_FIN:
    fecha_str = fecha_actual.strftime('%Y-%m-%d')

    try:
        response = requests.get(
            f'https://www.sofascore.com/api/v1/unique-tournament/19039/scheduled-events/{fecha_str}',
            impersonate="chrome120",
            timeout=10
        )

        if response.status_code == 200:
            lista_partidos = response.json().get('events', [])

            if lista_partidos:
                df_dia = pd.json_normalize(lista_partidos)
                df_dia['torneo_id']     = 19039
                df_dia['torneo_nombre'] = 'Czech Liga Pro'
                df_dia = limpiar_dataframe_para_sql(df_dia)
                guardados  = guardar_sin_duplicados(df_dia, conexion)
                duplicados = len(df_dia) - guardados

                total_guardados  += guardados
                total_duplicados += duplicados
                dias_con_datos   += 1

                print(f"  {fecha_str} -> {guardados} nuevos | {duplicados} duplicados | acumulado: {total_guardados}")
            else:
                dias_sin_datos += 1
                print(f"  {fecha_str} -> Sin partidos")

        elif response.status_code == 304:
            dias_sin_datos += 1
            print(f"  {fecha_str} -> Sin cambios (304)")

        else:
            errores += 1
            print(f"  {fecha_str} -> Error {response.status_code}")

    except Exception as e:
        errores += 1
        print(f"  {fecha_str} -> Excepcion: {str(e)[:50]}")

    time.sleep(0.5)
    fecha_actual += timedelta(days=1)

conexion.close()

print(f"\n{'='*55}")
print("REPORTE EXTRACCION HISTORICA")
print("="*55)
print(f"  Torneo             : Czech Liga Pro [19039]")
print(f"  Dias procesados    : {total_dias}")
print(f"  Dias con partidos  : {dias_con_datos}")
print(f"  Dias sin partidos  : {dias_sin_datos}")
print(f"  Partidos guardados : {total_guardados}")
print(f"  Duplicados ignored : {total_duplicados}")
print(f"  Errores            : {errores}")
print("="*55)
print("\nhistorical_extractor.py ejecutado correctamente")