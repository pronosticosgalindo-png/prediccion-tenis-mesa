import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd

print("="*55)
print("VERIFICACION DE DATOS DISPONIBLES")
print("="*55)

conexion = sqlite3.connect("base_ping_pong.sqlite")
df = pd.read_sql("SELECT * FROM matches_finalizados", conexion)
conexion.close()

# ── 1. RESULTADO EXACTO DE SETS ────────────────────────────
print("\n1. RESULTADO EXACTO (3-0, 3-1, 3-2):")
df['resultado_sets'] = df['sets_local'].astype(str) + "-" + df['sets_visitante'].astype(str)
dist = df['resultado_sets'].value_counts()
for res, cnt in dist.items():
    print(f"   {res}: {cnt} partidos ({cnt/len(df)*100:.1f}%)")

# ── 2. GANADOR SET 1 ───────────────────────────────────────
print("\n2. GANADOR SET 1 vs GANADOR PARTIDO:")
df['gano_set1'] = pd.to_numeric(df['puntos_local_s1'], errors='coerce') > pd.to_numeric(df['puntos_visitante_s1'], errors='coerce')
df['gano_partido_local'] = df['ganador'] == df['jugador_local']
df_valido = df.dropna(subset=['puntos_local_s1', 'puntos_visitante_s1'])
coincide = (df_valido['gano_set1'] == df_valido['gano_partido_local']).sum()
print(f"   Quien gana Set 1 gana el partido: {coincide/len(df_valido)*100:.1f}%")

# ── 3. HORARIO ─────────────────────────────────────────────
print("\n3. DATOS DE HORARIO:")
df['hora'] = pd.to_datetime(df['fecha']).dt.hour
print(f"   Partidos con hora disponible: {df['hora'].notna().sum()}")
print(f"   Rango de horas: {df['hora'].min():.0f}h - {df['hora'].max():.0f}h")

# ── 4. COLUMNAS DISPONIBLES ────────────────────────────────
print("\n4. COLUMNAS DE PUNTOS DISPONIBLES:")
for i in range(1, 6):
    col_l = f'puntos_local_s{i}'
    col_v = f'puntos_visitante_s{i}'
    if col_l in df.columns:
        validos = df[col_l].notna().sum()
        print(f"   Set {i}: {validos} partidos con datos")

print("\nverificar_datos.py ejecutado correctamente")