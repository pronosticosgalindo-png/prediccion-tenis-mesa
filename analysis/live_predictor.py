import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
import numpy as np

print("="*55)
print("PREDICTOR EN VIVO - RUSHBET")
print("="*55)

# ── 1. CARGAR DATOS ────────────────────────────────────────
conexion = sqlite3.connect("base_ping_pong.sqlite")
df       = pd.read_sql("SELECT * FROM matches_finalizados", conexion)
features = pd.read_sql("SELECT * FROM player_features", conexion)
h2h      = pd.read_sql("SELECT * FROM head_to_head", conexion)
conexion.close()

df = df[df['total_sets'] > 0].copy()

for i in range(1, 6):
    col_l = f'puntos_local_s{i}'
    col_v = f'puntos_visitante_s{i}'
    if col_l in df.columns and col_v in df.columns:
        df[f'puntos_set{i}'] = (
            pd.to_numeric(df[col_l], errors='coerce') +
            pd.to_numeric(df[col_v], errors='coerce')
        )

# ── 2. DATOS DEL PARTIDO EN VIVO ───────────────────────────
# MODIFICA ESTOS DATOS SEGUN LO QUE VES EN RUSHBET
LOCAL      = "Postelt V."
VISITANTE  = "Parhomenko D."
SET_ACTUAL = 4
MARC_LOCAL = 0
MARC_VISIT = 0

# Puntos totales de cada set ya jugado
SETS_ANTERIORES = [17, 18, 20]

# ── 3. FUNCIONES ───────────────────────────────────────────
def get_stats(nombre, features_df):
    row = features_df[features_df['jugador'] == nombre]
    if len(row) == 0:
        return {'win_rate': 0.5, 'sets_ratio': 0.5,
                'victorias_ultimos_5': 0, 'partidos': 0}
    r = row.iloc[0]
    return {
        'win_rate'            : r['win_rate'],
        'sets_ratio'          : r['sets_ratio'],
        'victorias_ultimos_5' : r['victorias_ultimos_5'],
        'partidos'            : r['partidos'],
    }

def get_h2h_stats(jugador_a, jugador_b, h2h_df):
    mask = (
        ((h2h_df['jugador_a'] == jugador_a) & (h2h_df['jugador_b'] == jugador_b)) |
        ((h2h_df['jugador_a'] == jugador_b) & (h2h_df['jugador_b'] == jugador_a))
    )
    fila = h2h_df[mask]
    if len(fila) == 0:
        return 0.5, 0
    f     = fila.iloc[0]
    total = f['partidos']
    if f['jugador_a'] == jugador_a:
        return round(f['victorias_a'] / total, 3), total
    else:
        return round(f['victorias_b'] / total, 3), total

def prob_linea(datos, linea):
    if datos is None or len(datos) == 0:
        return 50.0, 50.0
    over  = round((datos > linea).sum() / len(datos) * 100, 1)
    under = round((datos <= linea).sum() / len(datos) * 100, 1)
    return over, under

def recomendacion(prob_over, prob_under, linea):
    if prob_over >= 65:
        return f"APOSTAR MAS DE {linea} ({prob_over}%)"
    elif prob_under >= 65:
        return f"APOSTAR MENOS DE {linea} ({prob_under}%)"
    else:
        return f"NO APOSTAR - parejo ({prob_over}% vs {prob_under}%)"

# ── 4. INFORMACION DEL PARTIDO ─────────────────────────────
fl = get_stats(LOCAL, features)
fv = get_stats(VISITANTE, features)
h2h_score, h2h_partidos = get_h2h_stats(LOCAL, VISITANTE, h2h)

print(f"\nPartido  : {LOCAL} vs {VISITANTE}")
print(f"Set      : {SET_ACTUAL} | Marcador: {MARC_LOCAL}-{MARC_VISIT}")
print(f"\nHistorial jugadores:")
print(f"  {LOCAL:<20} Win rate: {fl['win_rate']*100:.1f}% | Sets ratio: {fl['sets_ratio']*100:.1f}%")
print(f"  {VISITANTE:<20} Win rate: {fv['win_rate']*100:.1f}% | Sets ratio: {fv['sets_ratio']*100:.1f}%")
print(f"  H2H: {h2h_partidos} partidos | Ventaja local: {h2h_score*100:.1f}%")

# ── 5. PATRON DE SETS ANTERIORES ───────────────────────────
if SETS_ANTERIORES:
    print(f"\n{'='*55}")
    print(f"PATRON DE SETS ANTERIORES")
    print(f"{'='*55}")

    for idx, pts in enumerate(SETS_ANTERIORES, 1):
        nivel = "ALTO" if pts > 19.5 else "MEDIO" if pts >= 17.5 else "BAJO"
        print(f"  Set {idx}: {pts} puntos [{nivel}]")

    promedio_partido = sum(SETS_ANTERIORES) / len(SETS_ANTERIORES)
    ultimo_set       = SETS_ANTERIORES[-1]
    tendencia        = "SUBIENDO" if SETS_ANTERIORES[-1] > SETS_ANTERIORES[0] else "BAJANDO"

    print(f"\n  Promedio del partido : {promedio_partido:.1f} puntos")
    print(f"  Tendencia            : {tendencia}")

    # Patron historico: despues de un set alto/bajo que sigue
    col_set_ant = f'puntos_set{SET_ACTUAL - 1}'
    col_set_act = f'puntos_set{SET_ACTUAL}'

    if col_set_ant in df.columns and col_set_act in df.columns:
        df_valido = df[[col_set_ant, col_set_act]].dropna()

        # Partidos donde el set anterior fue similar al ultimo set jugado
        margen = 2
        df_similar = df_valido[
            (df_valido[col_set_ant] >= ultimo_set - margen) &
            (df_valido[col_set_ant] <= ultimo_set + margen)
        ][col_set_act]

        if len(df_similar) >= 10:
            print(f"\n  Cuando el set anterior tuvo ~{ultimo_set} puntos:")
            print(f"  El siguiente set promedió: {df_similar.mean():.1f} puntos")
            print(f"  Min: {df_similar.min():.0f} | Max: {df_similar.max():.0f}")

            for linea in [17.5, 18.5, 19.5]:
                over, under = prob_linea(df_similar, linea)
                rec = recomendacion(over, under, linea)
                print(f"\n  Linea {linea}: Over={over}% | Under={under}%")
                print(f"  -> {rec}")

# ── 6. ANALISIS DEL SET ACTUAL ─────────────────────────────
print(f"\n{'='*55}")
print(f"ANALISIS SET {SET_ACTUAL} - APUESTAS EN VIVO")
print(f"{'='*55}")

puntos_actuales = MARC_LOCAL + MARC_VISIT
col_set = f'puntos_set{SET_ACTUAL}'

# Datos H2H del set actual
mask_h2h = (
    ((df['jugador_local'] == LOCAL) & (df['jugador_visitante'] == VISITANTE)) |
    ((df['jugador_local'] == VISITANTE) & (df['jugador_visitante'] == LOCAL))
)
df_h2h_set = df[mask_h2h][col_set].dropna() if col_set in df.columns else pd.Series()

# Datos generales del set
mask_local = (df['jugador_local'] == LOCAL) | (df['jugador_visitante'] == LOCAL)
mask_visit = (df['jugador_local'] == VISITANTE) | (df['jugador_visitante'] == VISITANTE)
df_jugadores = df[mask_local | mask_visit][col_set].dropna() if col_set in df.columns else pd.Series()
df_general   = df[col_set].dropna() if col_set in df.columns else pd.Series()

print(f"\n  Puntos actuales en set {SET_ACTUAL}: {puntos_actuales}")

for linea in [17.5, 18.5, 19.5]:
    restantes = linea - puntos_actuales
    print(f"\n  {'='*45}")
    print(f"  LINEA {linea} puntos")

    if restantes <= 0:
        print(f"  LINEA YA SUPERADA (van {puntos_actuales} puntos)")
        continue

    print(f"  Faltan {restantes:.1f} puntos para superar la linea")

    if len(df_h2h_set) >= 3:
        over, under = prob_linea(df_h2h_set, linea)
        print(f"  H2H directo ({len(df_h2h_set)} partidos): Over={over}% | Under={under}%")
        datos_ref = df_h2h_set
    elif len(df_jugadores) >= 5:
        over, under = prob_linea(df_jugadores, linea)
        print(f"  Historico jugadores ({len(df_jugadores)} sets): Over={over}% | Under={under}%")
        datos_ref = df_jugadores
    else:
        over, under = prob_linea(df_general, linea)
        print(f"  Historico general ({len(df_general)} sets): Over={over}% | Under={under}%")
        datos_ref = df_general

    print(f"  -> {recomendacion(over, under, linea)}")

# ── 7. ALERTA DE MARCADOR ──────────────────────────────────
print(f"\n{'='*55}")
print(f"ALERTA DE MARCADOR")
print(f"{'='*55}")

diferencia = abs(MARC_LOCAL - MARC_VISIT)
if diferencia <= 2 and puntos_actuales >= 18:
    print(f"  ALERTA: Marcador reñido ({MARC_LOCAL}-{MARC_VISIT}) con {puntos_actuales} puntos")
    print(f"  Probable DEUCE -> set puede llegar a 22-24 puntos")
    print(f"  -> APOSTAR MAS DE 19.5 si aun esta disponible")
elif diferencia >= 4:
    lider = LOCAL if MARC_LOCAL > MARC_VISIT else VISITANTE
    print(f"  {lider} domina por {diferencia} puntos")
    print(f"  Probable cierre rapido -> MENOS puntos")
    print(f"  -> APOSTAR MENOS DE 18.5 o 19.5")
else:
    print(f"  Partido equilibrado - seguir el historico")

print(f"\n{'='*55}")
print("live_predictor.py ejecutado correctamente")