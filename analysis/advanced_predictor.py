import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
import numpy as np
import pickle

print("="*55)
print("PREDICTOR AVANZADO - CZECH LIGA PRO")
print("="*55)

# ── 1. CARGAR MODELOS Y DATOS ──────────────────────────────
with open('analysis/modelo_avanzado.pkl', 'rb') as f:
    paquete = pickle.load(f)

modelos       = paquete['modelos']
le_res        = paquete['le_res']
le_sets       = paquete['le_sets']
feature_cols  = paquete['feature_cols']
accuracies    = paquete['accuracies']

conexion  = sqlite3.connect("base_ping_pong.sqlite")
features  = pd.read_sql("SELECT * FROM player_features", conexion)
h2h       = pd.read_sql("SELECT * FROM head_to_head", conexion)
df_hist   = pd.read_sql("SELECT * FROM matches_finalizados", conexion)
pendientes = pd.read_sql("SELECT * FROM matches_pendientes", conexion)
conexion.close()

df_hist['fecha'] = pd.to_datetime(df_hist['fecha'])
df_hist['hora']  = df_hist['fecha'].dt.hour

print(f"Modelos cargados     : {len(modelos)}")
print(f"Jugadores en base    : {len(features)}")
print(f"Partidos pendientes  : {len(pendientes)}")

# ── 2. FUNCIONES ───────────────────────────────────────────
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

def get_h2h_score(jugador_a, jugador_b, h2h_df):
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

def get_wr_horario(jugador, hora, df):
    franja = 'manana' if hora < 12 else 'tarde' if hora < 18 else 'noche'
    if franja == 'manana':   mask_hora = df['hora'] < 12
    elif franja == 'tarde':  mask_hora = (df['hora'] >= 12) & (df['hora'] < 18)
    else:                    mask_hora = df['hora'] >= 18

    mask_jug = (df['jugador_local'] == jugador) | (df['jugador_visitante'] == jugador)
    df_jug   = df[mask_jug & mask_hora]

    if len(df_jug) == 0:
        return 0.5
    return round((df_jug['ganador'] == jugador).sum() / len(df_jug), 3)

def construir_features(local, visitante, hora=15):
    fl = get_stats(local, features)
    fv = get_stats(visitante, features)
    h2h_score, h2h_n = get_h2h_score(local, visitante, h2h)
    wr_l_hora = get_wr_horario(local, hora, df_hist)
    wr_v_hora = get_wr_horario(visitante, hora, df_hist)

    return pd.DataFrame([{
        'local_win_rate'   : fl['win_rate'],
        'local_sets_ratio' : fl['sets_ratio'],
        'local_racha'      : fl['victorias_ultimos_5'],
        'local_partidos'   : fl['partidos'],
        'visit_win_rate'   : fv['win_rate'],
        'visit_sets_ratio' : fv['sets_ratio'],
        'visit_racha'      : fv['victorias_ultimos_5'],
        'visit_partidos'   : fv['partidos'],
        'diff_win_rate'    : fl['win_rate']   - fv['win_rate'],
        'diff_sets_ratio'  : fl['sets_ratio'] - fv['sets_ratio'],
        'diff_racha'       : fl['victorias_ultimos_5'] - fv['victorias_ultimos_5'],
        'h2h_local'        : h2h_score,
        'h2h_partidos'     : h2h_n,
        'hora'             : hora,
        'wr_local_hora'    : wr_l_hora,
        'wr_visit_hora'    : wr_v_hora,
        'diff_wr_hora'     : wr_l_hora - wr_v_hora,
    }])

def nivel(p):
    if p >= 80: return "[ALTA]"
    if p >= 65: return "[MEDIA]"
    return "[BAJA]"

def predecir_completo(local, visitante, hora=15):
    fl = get_stats(local, features)
    fv = get_stats(visitante, features)
    h2h_score, h2h_n = get_h2h_score(local, visitante, h2h)

    X = construir_features(local, visitante, hora)

    # Modelo 1 - Ganador partido
    prob_gana = modelos['ganador_partido'].predict_proba(X)[0]
    prob_local    = round(prob_gana[1] * 100, 1)
    prob_visitante = round(prob_gana[0] * 100, 1)
    ganador = local if prob_local > prob_visitante else visitante
    conf_ganador = max(prob_local, prob_visitante)

    # Modelo 2 - Ganador Set 1
    prob_set1 = modelos['ganador_set1'].predict_proba(X)[0]
    prob_set1_local    = round(prob_set1[1] * 100, 1)
    prob_set1_visitante = round(prob_set1[0] * 100, 1)
    ganador_set1 = local if prob_set1_local > prob_set1_visitante else visitante
    conf_set1    = max(prob_set1_local, prob_set1_visitante)

    # Modelo 3 - Resultado exacto
    prob_res  = modelos['resultado_exacto'].predict_proba(X)[0]
    clases_res = le_res.classes_
    idx_max   = np.argmax(prob_res)
    resultado_pred = clases_res[idx_max]
    conf_res  = round(prob_res[idx_max] * 100, 1)

    # Top 3 resultados
    top3_idx = np.argsort(prob_res)[::-1][:3]
    top3_res = [(clases_res[i], round(prob_res[i]*100, 1)) for i in top3_idx]

    # Modelo 4 - Total sets
    prob_sets  = modelos['total_sets'].predict_proba(X)[0]
    clases_sets = le_sets.classes_
    idx_sets   = np.argmax(prob_sets)
    sets_pred  = clases_sets[idx_sets]
    conf_sets  = round(prob_sets[idx_sets] * 100, 1)

    # Modelo 5 - Puntos totales
    puntos_pred = round(modelos['puntos_totales'].predict(X)[0], 1)

    # Horario
    franja = 'Manana' if hora < 12 else 'Tarde' if hora < 18 else 'Noche'
    wr_l   = get_wr_horario(local, hora, df_hist)
    wr_v   = get_wr_horario(visitante, hora, df_hist)

    return {
        'local'             : local,
        'visitante'         : visitante,
        'hora'              : hora,
        'franja'            : franja,
        'ganador'           : ganador,
        'prob_local'        : prob_local,
        'prob_visitante'    : prob_visitante,
        'conf_ganador'      : conf_ganador,
        'ganador_set1'      : ganador_set1,
        'prob_set1_local'   : prob_set1_local,
        'prob_set1_visit'   : prob_set1_visitante,
        'conf_set1'         : conf_set1,
        'resultado_pred'    : resultado_pred,
        'conf_res'          : conf_res,
        'top3_res'          : top3_res,
        'sets_pred'         : sets_pred,
        'conf_sets'         : conf_sets,
        'puntos_pred'       : puntos_pred,
        'h2h_partidos'      : h2h_n,
        'h2h_score'         : round(h2h_score * 100, 1),
        'wr_local'          : round(fl['win_rate'] * 100, 1),
        'wr_visitante'      : round(fv['win_rate'] * 100, 1),
        'wr_local_hora'     : round(wr_l * 100, 1),
        'wr_visit_hora'     : round(wr_v * 100, 1),
    }

def mostrar_prediccion(r):
    print(f"\n{'='*55}")
    print(f"  {r['local']} vs {r['visitante']}")
    print(f"  Hora: {r['hora']}h ({r['franja']})")
    print(f"{'='*55}")

    print(f"\n  GANADOR DEL PARTIDO")
    print(f"  {'-'*45}")
    print(f"  Prediccion : {r['ganador']}")
    print(f"  Prob       : {r['local']} {r['prob_local']}% - {r['visitante']} {r['prob_visitante']}%")
    print(f"  Confianza  : {r['conf_ganador']}% {nivel(r['conf_ganador'])}")
    print(f"  Win rate   : {r['local']} {r['wr_local']}% - {r['visitante']} {r['wr_visitante']}%")
    print(f"  H2H        : {r['h2h_partidos']} partidos | Local gana {r['h2h_score']}%")

    print(f"\n  GANADOR SET 1")
    print(f"  {'-'*45}")
    print(f"  Prediccion : {r['ganador_set1']}")
    print(f"  Prob       : {r['local']} {r['prob_set1_local']}% - {r['visitante']} {r['prob_set1_visit']}%")
    print(f"  Confianza  : {r['conf_set1']}% {nivel(r['conf_set1'])}")

    print(f"\n  RESULTADO EXACTO DE SETS")
    print(f"  {'-'*45}")
    print(f"  Mas probable: {r['resultado_pred']} ({r['conf_res']}%)")
    print(f"  Top 3 resultados:")
    for res, prob in r['top3_res']:
        print(f"    {res}: {prob}%")

    print(f"\n  TOTAL SETS Y PUNTOS")
    print(f"  {'-'*45}")
    print(f"  Sets esperados : {r['sets_pred']} ({r['conf_sets']}%)")
    print(f"  Puntos totales : ~{r['puntos_pred']} puntos")
    print(f"  Puntos por set : ~{round(r['puntos_pred']/int(r['sets_pred'][0]), 1)} puntos/set")

    print(f"\n  RENDIMIENTO POR HORARIO ({r['franja']})")
    print(f"  {'-'*45}")
    print(f"  {r['local']:<20} : {r['wr_local_hora']}% win rate en esta franja")
    print(f"  {r['visitante']:<20} : {r['wr_visit_hora']}% win rate en esta franja")

# ── 3. PARTIDOS DE RUSHBET ─────────────────────────────────
print(f"\n{'='*55}")
print("PREDICCIONES RUSHBET - HOY Y MANANA")
print("="*55)

# MODIFICA ESTOS PARTIDOS CON LOS DE RUSHBET
partidos_rushbet = [
    ("Jiri Plachy",          "Darin K.",      21),
    ("Huk M.",               "Svacha O.",     21),
    ("Sychra M.",            "Kasnik S.",     21),
    ("Jiri Grohsgott",       "Regner M.",     21),
    ("Zuzanek J.",           "Kolisnyk O.",   1),
    ("Kostal M.",            "Havlicek V.",   1),
    ("Wawrosz P.",           "Navedla M.",    1),
    ("Zientek Z.",           "Varecha P.",    1),
    ("Vavrecka M.",          "Prikasky L.",   1),
    ("Pospisil M.",          "Belovsky J.",   1),
    ("Kowolowski M.",        "Zurek F.",      1),
    ("Hruska Snr V.",        "Madle L.",      1),
    ("Vejvoda V.",           "Klement M.",    1),
    ("Navedla M.",           "Vavrecka M.",   2),
    ("Regner M.",            "Hruska Snr V.", 2),
    ("Kolisnyk O.",          "Vejvoda V.",    2),
]

jugadores_base = set(features['jugador'].tolist())
predicciones   = []
sin_historial  = []

for local, visitante, hora in partidos_rushbet:
    if local in jugadores_base and visitante in jugadores_base:
        r = predecir_completo(local, visitante, hora)
        mostrar_prediccion(r)
        predicciones.append(r)
    else:
        sin_historial.append((local, visitante))

if sin_historial:
    print(f"\n{'='*55}")
    print("SIN HISTORIAL EN LA BASE:")
    for l, v in sin_historial:
        print(f"  -> {l} vs {v}")

# ── 4. RESUMEN APUESTAS RECOMENDADAS ──────────────────────
print(f"\n{'='*55}")
print("RESUMEN - APUESTAS RECOMENDADAS RUSHBET")
print("="*55)

print(f"\n  GANADOR [ALTA confianza 80%+]:")
altas = [r for r in predicciones if r['conf_ganador'] >= 80]
if altas:
    for r in sorted(altas, key=lambda x: x['conf_ganador'], reverse=True):
        print(f"  -> {r['ganador']:<20} ({r['conf_ganador']}%) | {r['local']} vs {r['visitante']}")
else:
    print("  Ninguna con confianza ALTA hoy")

print(f"\n  GANADOR [MEDIA confianza 65-79%]:")
medias = [r for r in predicciones if 65 <= r['conf_ganador'] < 80]
if medias:
    for r in sorted(medias, key=lambda x: x['conf_ganador'], reverse=True):
        print(f"  -> {r['ganador']:<20} ({r['conf_ganador']}%) | {r['local']} vs {r['visitante']}")
else:
    print("  Ninguna con confianza MEDIA hoy")

print(f"\n{'='*55}")
print("advanced_predictor.py ejecutado correctamente")