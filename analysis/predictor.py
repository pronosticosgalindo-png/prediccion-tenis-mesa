import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
import numpy as np
import pickle

print("Sistema de Prediccion - Czech Liga Pro")
print("="*55)

# ── 1. CARGAR MODELO Y DATOS ───────────────────────────────
with open('analysis/modelo_ping_pong.pkl', 'rb') as f:
    paquete = pickle.load(f)

modelo        = paquete['modelo']
feature_cols  = paquete['feature_cols']
modelo_nombre = paquete['nombre']
accuracy      = paquete['accuracy']

conexion   = sqlite3.connect("base_ping_pong.sqlite")
features   = pd.read_sql("SELECT * FROM player_features", conexion)
h2h        = pd.read_sql("SELECT * FROM head_to_head", conexion)
pendientes = pd.read_sql("SELECT * FROM matches_pendientes", conexion)
conexion.close()

print(f"Modelo cargado     : {modelo_nombre}")
print(f"Accuracy historico : {accuracy:.3f}")
print(f"Jugadores en base  : {len(features)}")
print(f"Partidos pendientes: {len(pendientes)}")

# ── 2. FUNCIONES DE SOPORTE ───────────────────────────────
def get_features_jugador(nombre, features_df):
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

def get_h2h(jugador_a, jugador_b, h2h_df):
    mask = (
        ((h2h_df['jugador_a'] == jugador_a) & (h2h_df['jugador_b'] == jugador_b)) |
        ((h2h_df['jugador_a'] == jugador_b) & (h2h_df['jugador_b'] == jugador_a))
    )
    fila = h2h_df[mask]
    if len(fila) == 0:
        return 0.5
    f = fila.iloc[0]
    total = f['partidos']
    if f['jugador_a'] == jugador_a:
        return round(f['victorias_a'] / total, 3)
    else:
        return round(f['victorias_b'] / total, 3)

def predecir_partido(local, visitante, features, h2h):
    fl        = get_features_jugador(local, features)
    fv        = get_features_jugador(visitante, features)
    h2h_score = get_h2h(local, visitante, h2h)

    X = pd.DataFrame([{
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
    }])

    prob = modelo.predict_proba(X)[0]
    pred = modelo.predict(X)[0]

    return {
        'local'          : local,
        'visitante'      : visitante,
        'prob_local'     : round(prob[1] * 100, 1),
        'prob_visitante' : round(prob[0] * 100, 1),
        'prediccion'     : local if pred == 1 else visitante,
        'confianza'      : round(max(prob) * 100, 1),
        'h2h_local'      : round(h2h_score * 100, 1),
        'win_rate_local' : round(fl['win_rate'] * 100, 1),
        'win_rate_visit' : round(fv['win_rate'] * 100, 1),
    }

def nivel_confianza(confianza):
    if confianza >= 80: return "[ALTA]"
    if confianza >= 60: return "[MEDIA]"
    return "[BAJA]"

# ── 3. PREDICCIONES AUTOMATICAS DE PENDIENTES ─────────────
print(f"\n{'='*55}")
print(f"PARTIDOS PENDIENTES: {len(pendientes)}")
print("="*55)

predicciones = []

if len(pendientes) > 0:
    for _, p in pendientes.iterrows():
        local     = p.get('jugador_local', '')
        visitante = p.get('jugador_visitante', '')
        fecha     = p.get('fecha', 'Sin fecha')

        if not local or not visitante:
            continue

        r = predecir_partido(local, visitante, features, h2h)
        r['fecha'] = fecha
        predicciones.append(r)
else:
    print("  No hay partidos pendientes para hoy.")

# ── 4. PREDICCIONES RUSHBET ────────────────────────────────
print(f"\n{'='*55}")
print("PREDICCIONES RUSHBET - PARTIDOS DE HOY Y MANANA")
print("="*55)

jugadores_disponibles = set(features['jugador'].tolist())

# ── ACTUALIZA ESTOS PARTIDOS CADA DIA DESDE RUSHBET ───────
partidos_rushbet = [
    ("Jiri Plachy",      "Darin K."),
    ("Huk M.",           "Svacha O."),
    ("Sychra M.",        "Kasnik S."),
    ("Jiri Grohsgott",   "Regner M."),
    ("Zuzanek J.",       "Kolisnyk O."),
    ("Kostal M.",        "Havlicek V."),
    ("Wawrosz P.",       "Navedla M."),
    ("Zientek Z.",       "Varecha P."),
    ("Vavrecka M.",      "Prikasky L."),
    ("Pospisil M.",      "Belovsky J."),
    ("Kowolowski M.",    "Zurek F."),
    ("Hruska Snr V.",    "Madle L."),
    ("Vejvoda V.",       "Klement M."),
    ("Navedla M.",       "Vavrecka M."),
    ("Regner M.",        "Hruska Snr V."),
    ("Kolisnyk O.",      "Vejvoda V."),
]

encontrados    = 0
no_encontrados = []

for local, visitante in partidos_rushbet:
    en_base_local     = local in jugadores_disponibles
    en_base_visitante = visitante in jugadores_disponibles

    if en_base_local and en_base_visitante:
        encontrados += 1
        r = predecir_partido(local, visitante, features, h2h)
        print(f"\n  {local} vs {visitante}")
        print(f"  Prediccion : {r['prediccion']}")
        print(f"  Prob       : {local} {r['prob_local']}% - {visitante} {r['prob_visitante']}%")
        print(f"  Confianza  : {r['confianza']}% {nivel_confianza(r['confianza'])}")
        print(f"  H2H local  : {r['h2h_local']}%")
        print(f"  Win rate   : {local} {r['win_rate_local']}% - {visitante} {r['win_rate_visit']}%")
        print(f"  {'-'*45}")
    else:
        no_encontrados.append((local, visitante))

if no_encontrados:
    print(f"\n  Jugadores sin historial en la base:")
    for local, visitante in no_encontrados:
        print(f"  -> {local} vs {visitante}")

print(f"\n  Partidos predichos    : {encontrados}/{len(partidos_rushbet)}")
print(f"  Partidos sin historial: {len(no_encontrados)}/{len(partidos_rushbet)}")

# ── 5. GUARDAR PREDICCIONES ───────────────────────────────
if predicciones:
    df_pred = pd.DataFrame(predicciones)
    conexion = sqlite3.connect("base_ping_pong.sqlite")
    df_pred.to_sql('predicciones', conexion, if_exists='replace', index=False)
    conexion.close()
    print(f"\n{len(predicciones)} predicciones guardadas en SQLite")

print(f"\n{'='*55}")
print("predictor.py ejecutado correctamente")