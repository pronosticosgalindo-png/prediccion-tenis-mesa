import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
import numpy as np

print("Iniciando Feature Engineering...")
print("="*55)

# ── 1. CARGAR DATOS ────────────────────────────────────────
conexion = sqlite3.connect("base_ping_pong.sqlite")
df = pd.read_sql("SELECT * FROM matches_finalizados", conexion)
print(f"Partidos finalizados cargados: {len(df)}")

df['fecha'] = pd.to_datetime(df['fecha'])

# ── 2. TABLA UNIFICADA POR JUGADOR ─────────────────────────
local = df[['match_id', 'fecha', 'jugador_local', 'ganador']].copy()
local.columns = ['match_id', 'fecha', 'jugador', 'ganador']
local['rol'] = 'local'

visitante = df[['match_id', 'fecha', 'jugador_visitante', 'ganador']].copy()
visitante.columns = ['match_id', 'fecha', 'jugador', 'ganador']
visitante['rol'] = 'visitante'

todos = pd.concat([local, visitante], ignore_index=True)
todos['victoria'] = (todos['jugador'] == todos['ganador']).astype(int)

# ── 3. WIN RATE ────────────────────────────────────────────
win_rate = (
    todos.groupby('jugador')['victoria']
    .agg(partidos='count', victorias='sum')
    .reset_index()
)
win_rate['win_rate'] = (win_rate['victorias'] / win_rate['partidos']).round(3)
win_rate.sort_values('win_rate', ascending=False, inplace=True)
print(f"\nWIN RATE - Top 10 jugadores:")
print(win_rate.head(10).to_string(index=False))

# ── 4. SETS RATIO ──────────────────────────────────────────
sets_local = df[['jugador_local', 'sets_local', 'sets_visitante']].copy()
sets_local.columns = ['jugador', 'sets_favor', 'sets_contra']

sets_visit = df[['jugador_visitante', 'sets_visitante', 'sets_local']].copy()
sets_visit.columns = ['jugador', 'sets_favor', 'sets_contra']

sets_todos = pd.concat([sets_local, sets_visit], ignore_index=True)
sets_ratio = (
    sets_todos.groupby('jugador')[['sets_favor', 'sets_contra']]
    .sum()
    .reset_index()
)
sets_ratio['sets_ratio'] = (
    sets_ratio['sets_favor'] /
    (sets_ratio['sets_favor'] + sets_ratio['sets_contra'])
).round(3)
print(f"\nSETS RATIO - Top 10 jugadores:")
print(sets_ratio.sort_values('sets_ratio', ascending=False).head(10).to_string(index=False))

# ── 5. RACHA RECIENTE (ultimos 5) ─────────────────────────
def racha_reciente(jugador, df_todos, n=5):
    partidos_jugador = (
        df_todos[df_todos['jugador'] == jugador]
        .sort_values('fecha', ascending=False)
        .head(n)
    )
    return partidos_jugador['victoria'].sum()

jugadores_unicos = todos['jugador'].unique()
print(f"\nCalculando racha de {len(jugadores_unicos)} jugadores...")

racha = pd.DataFrame({
    'jugador': jugadores_unicos,
    'victorias_ultimos_5': [
        racha_reciente(j, todos) for j in jugadores_unicos
    ]
})
print(f"Racha calculada correctamente.")

# ── 6. HEAD TO HEAD ────────────────────────────────────────
print(f"\nCalculando head to head ({len(df)} partidos)...")

h2h_df = df[['jugador_local', 'jugador_visitante', 'ganador',
             'sets_local', 'sets_visitante', 'fecha']].copy()
h2h_df.columns = ['jugador_a', 'jugador_b', 'ganador', 'sets_a', 'sets_b', 'fecha']
h2h_df['gano_a'] = (h2h_df['ganador'] == h2h_df['jugador_a']).astype(int)

h2h_summary = (
    h2h_df.groupby(['jugador_a', 'jugador_b'])
    .agg(
        partidos    = ('gano_a', 'count'),
        victorias_a = ('gano_a', 'sum'),
    )
    .reset_index()
)
h2h_summary['victorias_b'] = h2h_summary['partidos'] - h2h_summary['victorias_a']
print(f"Head to head calculado: {len(h2h_summary)} cruces.")

# ── 7. TABLA MAESTRA DE FEATURES ──────────────────────────
features = win_rate.merge(sets_ratio[['jugador', 'sets_ratio']], on='jugador')
features = features.merge(racha, on='jugador')

print(f"\n{'='*55}")
print("TABLA MAESTRA DE FEATURES - Top 15:")
print("="*55)
print(features.head(15).to_string(index=False))

# ── 8. GUARDAR EN SQLITE ───────────────────────────────────
features.to_sql('player_features', conexion, if_exists='replace', index=False)
h2h_summary.to_sql('head_to_head', conexion, if_exists='replace', index=False)
conexion.close()

print(f"\n{'='*55}")
print("REPORTE FEATURE ENGINEERING")
print("="*55)
print(f"  Jugadores procesados : {len(features)}")
print(f"  Features por jugador : {len(features.columns)}")
print(f"  Cruces head-to-head  : {len(h2h_summary)}")
print(f"  Tablas guardadas     :")
print(f"    -> player_features")
print(f"    -> head_to_head")
print("="*55)
print("\nfeatures.py ejecutado correctamente")