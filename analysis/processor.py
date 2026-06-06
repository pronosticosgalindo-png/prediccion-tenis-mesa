import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
from datetime import datetime, timezone

print("Iniciando procesamiento de datos...")
print("="*55)

# ── 1. CARGAR DATOS CRUDOS ─────────────────────────────────
conexion = sqlite3.connect("base_ping_pong.sqlite")

df_czech = pd.read_sql("SELECT * FROM matches_czech", conexion)
print(f"Registros Czech Liga Pro cargados: {len(df_czech)}")

# ── 2. SELECCIONAR COLUMNAS UTILES ─────────────────────────
columnas_utiles = {
    'id'                    : 'match_id',
    'startTimestamp'        : 'fecha_timestamp',
    'status.type'           : 'estado',
    'status.description'    : 'estado_desc',
    'tournament.name'       : 'torneo',
    'roundInfo.round'       : 'ronda',
    'roundInfo.name'        : 'ronda_nombre',
    'venue.city.name'       : 'ciudad',
    'venue.stadium.name'    : 'estadio',
    'homeTeam.name'         : 'jugador_local',
    'homeTeam.country.name' : 'pais_local',
    'awayTeam.name'         : 'jugador_visitante',
    'awayTeam.country.name' : 'pais_visitante',
    'homeScore.current'     : 'sets_local',
    'awayScore.current'     : 'sets_visitante',
    'homeScore.period1'     : 'puntos_local_s1',
    'awayScore.period1'     : 'puntos_visitante_s1',
    'homeScore.period2'     : 'puntos_local_s2',
    'awayScore.period2'     : 'puntos_visitante_s2',
    'homeScore.period3'     : 'puntos_local_s3',
    'awayScore.period3'     : 'puntos_visitante_s3',
    'homeScore.period4'     : 'puntos_local_s4',
    'awayScore.period4'     : 'puntos_visitante_s4',
    'homeScore.period5'     : 'puntos_local_s5',
    'awayScore.period5'     : 'puntos_visitante_s5',
    'homeScore.period6'     : 'puntos_local_s6',
    'awayScore.period6'     : 'puntos_visitante_s6',
    'homeScore.period7'     : 'puntos_local_s7',
    'awayScore.period7'     : 'puntos_visitante_s7',
    'torneo_nombre'         : 'torneo_nombre',
}

columnas_presentes = {k: v for k, v in columnas_utiles.items() if k in df_czech.columns}
df_clean = df_czech[list(columnas_presentes.keys())].copy()
df_clean.rename(columns=columnas_presentes, inplace=True)
print(f"Columnas seleccionadas: {len(df_clean.columns)}")

# ── 3. CONVERTIR TIMESTAMPS ────────────────────────────────
def convertir_timestamp(ts):
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return None

df_clean['fecha'] = df_clean['fecha_timestamp'].apply(convertir_timestamp)
df_clean.drop(columns=['fecha_timestamp'], inplace=True)

# ── 4. CONVERTIR COLUMNAS NUMERICAS ───────────────────────
cols_numericas = [c for c in df_clean.columns if any(x in c for x in ['sets_', 'puntos_'])]
for col in cols_numericas:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# ── 5. ELIMINAR DUPLICADOS ─────────────────────────────────
antes = len(df_clean)
df_clean.drop_duplicates(subset='match_id', keep='last', inplace=True)
print(f"Duplicados eliminados: {antes - len(df_clean)}")

# ── 6. COLUMNA GANADOR ─────────────────────────────────────
def determinar_ganador(row):
    try:
        if row['sets_local'] > row['sets_visitante']:
            return row['jugador_local']
        elif row['sets_visitante'] > row['sets_local']:
            return row['jugador_visitante']
        else:
            return 'Sin resultado'
    except:
        return 'Sin resultado'

df_clean['ganador'] = df_clean.apply(determinar_ganador, axis=1)

# ── 7. TOTAL SETS JUGADOS ──────────────────────────────────
df_clean['total_sets'] = (
    df_clean['sets_local'].fillna(0) + df_clean['sets_visitante'].fillna(0)
)

# ── 8. FILTRAR FINALIZADOS Y PENDIENTES ───────────────────
df_finalizados = df_clean[df_clean['estado'] == 'finished'].copy()
df_pendientes  = df_clean[df_clean['estado'] != 'finished'].copy()

print(f"Partidos finalizados : {len(df_finalizados)}")
print(f"Partidos pendientes  : {len(df_pendientes)}")

# ── 9. GUARDAR TABLAS ──────────────────────────────────────
df_clean.to_sql('matches_clean', conexion,
                if_exists='replace', index=False)

df_finalizados.to_sql('matches_finalizados', conexion,
                      if_exists='replace', index=False)

df_pendientes.to_sql('matches_pendientes', conexion,
                     if_exists='replace', index=False)

conexion.close()

# ── 10. REPORTE FINAL ──────────────────────────────────────
print("\n" + "="*55)
print("REPORTE DE PROCESAMIENTO")
print("="*55)
print(f"  Total registros procesados : {len(df_clean)}")
print(f"  Partidos finalizados       : {len(df_finalizados)}")
print(f"  Partidos pendientes        : {len(df_pendientes)}")
print(f"  Columnas en tabla limpia   : {len(df_clean.columns)}")
print(f"  Tablas guardadas en SQLite :")
print(f"    -> matches_clean")
print(f"    -> matches_finalizados")
print(f"    -> matches_pendientes")
print("="*55)
print("\nprocessor.py ejecutado correctamente")