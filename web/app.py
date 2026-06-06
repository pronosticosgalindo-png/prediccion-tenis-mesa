import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, render_template, jsonify, request
import sqlite3
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# ── RUTA A LA BASE DE DATOS ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, '..', 'base_ping_pong.sqlite')
MOD_PATH = os.path.join(BASE_DIR, '..', 'analysis', 'modelo_avanzado.pkl')

# ── CARGAR MODELO ─────────────────────────────────────────
with open(MOD_PATH, 'rb') as f:
    paquete = pickle.load(f)

modelos      = paquete['modelos']
le_res       = paquete['le_res']
le_sets      = paquete['le_sets']
feature_cols = paquete['feature_cols']

# ── FUNCIONES DE SOPORTE ──────────────────────────────────
def get_conexion():
    return sqlite3.connect(DB_PATH)

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

def get_h2h(jugador_a, jugador_b, h2h_df):
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
    df['hora'] = pd.to_datetime(df['fecha']).dt.hour
    if hora < 12:   mask_hora = df['hora'] < 12
    elif hora < 18: mask_hora = (df['hora'] >= 12) & (df['hora'] < 18)
    else:           mask_hora = df['hora'] >= 18
    mask_jug = (df['jugador_local'] == jugador) | (df['jugador_visitante'] == jugador)
    df_jug   = df[mask_jug & mask_hora]
    if len(df_jug) == 0:
        return 0.5
    return round((df_jug['ganador'] == jugador).sum() / len(df_jug), 3)

def construir_features(local, visitante, hora, features, h2h):
    fl = get_stats(local, features)
    fv = get_stats(visitante, features)
    h2h_score, h2h_n = get_h2h(local, visitante, h2h)

    conn   = get_conexion()
    df_hist = pd.read_sql("SELECT * FROM matches_finalizados", conn)
    conn.close()

    wr_l = get_wr_horario(local, hora, df_hist)
    wr_v = get_wr_horario(visitante, hora, df_hist)

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
        'wr_local_hora'    : wr_l,
        'wr_visit_hora'    : wr_v,
        'diff_wr_hora'     : wr_l - wr_v,
    }])

# ── RUTAS ─────────────────────────────────────────────────
@app.route('/')
def index():
    conn     = get_conexion()
    features = pd.read_sql("SELECT * FROM player_features", conn)
    conn.close()

    stats = {
        'total_jugadores' : len(features),
        'fecha'           : datetime.now().strftime('%d/%m/%Y %H:%M'),
    }

    try:
        conn  = get_conexion()
        preds = pd.read_sql("SELECT * FROM predicciones", conn)
        conn.close()
        stats['total_predicciones'] = len(preds)
    except:
        stats['total_predicciones'] = 0

    try:
        conn     = get_conexion()
        df_match = pd.read_sql("SELECT COUNT(*) as n FROM matches_finalizados", conn)
        conn.close()
        stats['total_partidos'] = int(df_match.iloc[0]['n'])
    except:
        stats['total_partidos'] = 0

    return render_template('index.html', stats=stats)

@app.route('/api/jugadores')
def api_jugadores():
    conn     = get_conexion()
    features = pd.read_sql(
        "SELECT jugador, partidos, victorias, win_rate, sets_ratio, victorias_ultimos_5 FROM player_features ORDER BY win_rate DESC",
        conn
    )
    conn.close()
    return jsonify(features.to_dict(orient='records'))

@app.route('/api/predicciones')
def api_predicciones():
    try:
        conn  = get_conexion()
        preds = pd.read_sql("SELECT * FROM predicciones ORDER BY confianza DESC", conn)
        conn.close()
        return jsonify(preds.to_dict(orient='records'))
    except:
        return jsonify([])

@app.route('/api/predecir', methods=['POST'])
def api_predecir():
    data      = request.json
    local     = data.get('local')
    visitante = data.get('visitante')
    hora      = int(data.get('hora', 15))

    conn     = get_conexion()
    features = pd.read_sql("SELECT * FROM player_features", conn)
    h2h      = pd.read_sql("SELECT * FROM head_to_head", conn)
    conn.close()

    fl = get_stats(local, features)
    fv = get_stats(visitante, features)
    h2h_score, h2h_n = get_h2h(local, visitante, h2h)

    X = construir_features(local, visitante, hora, features, h2h)

    prob_gana  = modelos['ganador_partido'].predict_proba(X)[0]
    prob_set1  = modelos['ganador_set1'].predict_proba(X)[0]
    prob_res   = modelos['resultado_exacto'].predict_proba(X)[0]
    prob_sets  = modelos['total_sets'].predict_proba(X)[0]
    puntos_pred = round(modelos['puntos_totales'].predict(X)[0], 1)

    clases_res  = le_res.classes_
    clases_sets = le_sets.classes_
    top3_res    = sorted(
        zip(clases_res, [round(p*100,1) for p in prob_res]),
        key=lambda x: x[1], reverse=True
    )[:3]

    return jsonify({
        'local'            : local,
        'visitante'        : visitante,
        'prob_local'       : round(prob_gana[1] * 100, 1),
        'prob_visitante'   : round(prob_gana[0] * 100, 1),
        'ganador'          : local if prob_gana[1] > prob_gana[0] else visitante,
        'confianza'        : round(max(prob_gana) * 100, 1),
        'ganador_set1'     : local if prob_set1[1] > prob_set1[0] else visitante,
        'conf_set1'        : round(max(prob_set1) * 100, 1),
        'resultado_pred'   : clases_res[np.argmax(prob_res)],
        'top3_resultados'  : top3_res,
        'sets_pred'        : clases_sets[np.argmax(prob_sets)],
        'conf_sets'        : round(max(prob_sets) * 100, 1),
        'puntos_totales'   : puntos_pred,
        'puntos_por_set'   : round(puntos_pred / 4, 1),
        'h2h_partidos'     : h2h_n,
        'h2h_score'        : round(h2h_score * 100, 1),
        'wr_local'         : round(fl['win_rate'] * 100, 1),
        'wr_visitante'     : round(fv['win_rate'] * 100, 1),
    })

@app.route('/api/live', methods=['POST'])
def api_live():
    data       = request.json
    local      = data.get('local')
    visitante  = data.get('visitante')
    set_actual = int(data.get('set_actual', 1))
    marc_local = int(data.get('marc_local', 0))
    marc_visit = int(data.get('marc_visit', 0))
    sets_ant   = data.get('sets_anteriores', [])

    conn   = get_conexion()
    df_hist = pd.read_sql("SELECT * FROM matches_finalizados", conn)
    conn.close()

    for i in range(1, 6):
        col_l = f'puntos_local_s{i}'
        col_v = f'puntos_visitante_s{i}'
        if col_l in df_hist.columns and col_v in df_hist.columns:
            df_hist[f'puntos_set{i}'] = (
                pd.to_numeric(df_hist[col_l], errors='coerce') +
                pd.to_numeric(df_hist[col_v], errors='coerce')
            )

    col_set = f'puntos_set{set_actual}'
    mask_h2h = (
        ((df_hist['jugador_local'] == local) & (df_hist['jugador_visitante'] == visitante)) |
        ((df_hist['jugador_local'] == visitante) & (df_hist['jugador_visitante'] == local))
    )

    df_h2h_set = df_hist[mask_h2h][col_set].dropna() if col_set in df_hist.columns else pd.Series()
    df_general = df_hist[col_set].dropna() if col_set in df_hist.columns else pd.Series()

    puntos_actuales = marc_local + marc_visit
    recomendaciones = []

    for linea in [17.5, 18.5, 19.5]:
        restantes = linea - puntos_actuales
        if restantes <= 0:
            recomendaciones.append({
                'linea'  : linea,
                'estado' : 'SUPERADA',
                'mensaje': f'Linea ya superada (van {puntos_actuales} puntos)'
            })
            continue

        datos = df_h2h_set if len(df_h2h_set) >= 3 else df_general
        fuente = 'H2H' if len(df_h2h_set) >= 3 else 'General'

        over  = round((datos > linea).sum() / len(datos) * 100, 1) if len(datos) > 0 else 50
        under = round((datos <= linea).sum() / len(datos) * 100, 1) if len(datos) > 0 else 50

        if over >= 65:
            accion = f'APOSTAR MAS DE {linea}'
            color  = 'green'
        elif under >= 65:
            accion = f'APOSTAR MENOS DE {linea}'
            color  = 'green'
        else:
            accion = 'NO APOSTAR'
            color  = 'red'

        recomendaciones.append({
            'linea'    : linea,
            'over'     : over,
            'under'    : under,
            'accion'   : accion,
            'color'    : color,
            'fuente'   : fuente,
            'restantes': restantes,
        })

    diferencia = abs(marc_local - marc_visit)
    if diferencia <= 2 and puntos_actuales >= 18:
        alerta = 'REÑIDO - Probable deuce - MAS puntos'
    elif diferencia >= 4:
        alerta = 'Un jugador domina - Probable cierre rapido - MENOS puntos'
    else:
        alerta = 'Partido equilibrado - Seguir historico'

    return jsonify({
        'recomendaciones' : recomendaciones,
        'alerta'          : alerta,
        'puntos_actuales' : puntos_actuales,
        'sets_anteriores' : sets_ant,
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)