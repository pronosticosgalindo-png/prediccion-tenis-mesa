import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
import pickle
import numpy as np
import requests
import json
import os
from datetime import datetime
from curl_cffi import requests as curl_requests

# ── CONFIGURACION ─────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_DIR, 'base_ping_pong.sqlite')
MOD_PATH  = os.path.join(BASE_DIR, 'analysis', 'modelo_avanzado.pkl')

TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '8971209009:AAEG4VnJJnP4qwD3lUIQx_mLk2E6mOQkrOs')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '1546171444')

# ── TELEGRAM ──────────────────────────────────────────────
def enviar_telegram(mensaje):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': mensaje,
            'parse_mode': 'HTML'
        }, timeout=10)
        print(f"Telegram: {mensaje[:50]}...")
    except Exception as e:
        print(f"Error Telegram: {e}")

# ── EXTRACTOR ─────────────────────────────────────────────
def extraer_partidos():
    FECHA_HOY = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*50}")
    print(f"Extrayendo partidos: {FECHA_HOY}")

    cookies = {
        '_cc_id': '665c530e0f6aa71a8a4cb0637145eaa7',
        '_ga': 'GA1.1.1956304625.1762201631',
        'AD_VERGE_SESSION_COOKIE_V1': '8f5072f4-2b49-4f5c-a190-6e99e705eb06',
        '_adv_uid': 'c97e1656-b739-4520-ad16-137ebb86720a',
        'hb_insticator_uid': '847eaa92-6071-4fbc-bd86-227c62f6fd6d',
        '_gcl_au': '1.1.835639633.1780018185',
        '_adv_sid': '4e728cd8-f741-4726-81f3-5f49f9ef9cc7',
        'ssp_test': 'control',
        '_li_dcdm_c': '.sofascore.com',
        '__gads': 'ID=86ba3bf017fab095:T=1762201627:RT=1780863181:S=ALNI_MYXKVhFZ-tm3js_y8AGbLqMYgLU_A',
        '__gpi': 'UID=000012a59574e3c7:T=1762201627:RT=1780863181:S=ALNI_MYIUpKlhX_9Lvn6I8c9rYA1mKByZQ',
        '__eoi': 'ID=9f47ba8481905a21:T=1780018205:RT=1780863181:S=AA-AfjZK16GEX47BwSM7nZyc3Mj9',
        'FCCDCF': '%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%22398b09f5-8b5b-4d0d-8a97-49e5c52d93ce%5C%22%2C%5B1762201626%2C136000000%5D%5D%22%5D%5D%5D',
        'FCNEC': '%5B%5B%22AKsRol-BvZAfhBkdcnpR7305MbcOO2QCbXRtvQmAq6bv0zf7oAotcS9TyslRuvdLbfhzCcD8R48MUz70oR7C2M-346sgdfWqjDI79SElu5JStnIyziurNszcBRMYJ5qG-0Gvlor8Oszh-0w5PlFpBdhdxe8m9ZWEOA%3D%3D%22%5D%5D',
        '_ga_HNQ9P9MGZR': 'GS2.1.s1780863179$o20$g1$t1780863406$j60$l0$h0',
    }

    headers = {
        'accept': '*/*',
        'accept-language': 'es-ES,es;q=0.9',
        'referer': 'https://www.sofascore.com/es/table-tennis',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
    }

    try:
        response = curl_requests.get(
            f'https://www.sofascore.com/api/v1/unique-tournament/24047/scheduled-events/{FECHA_HOY}',
            cookies=cookies, headers=headers, impersonate="chrome120"
        )

        if response.status_code != 200:
            print(f"Error HTTP: {response.status_code}")
            return 0

        lista = response.json().get('events', [])
        if not lista:
            print("Sin partidos hoy")
            return 0

        df_nuevos = pd.json_normalize(lista)
        for col in df_nuevos.columns:
            if df_nuevos[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df_nuevos[col] = df_nuevos[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )

        conn = sqlite3.connect(DB_PATH)
        try:
            df_exist = pd.read_sql("SELECT id FROM matches", conn)
            ids_exist = set(df_exist['id'].astype(str).tolist())
            df_filtrado = df_nuevos[~df_nuevos['id'].astype(str).isin(ids_exist)]
        except:
            df_filtrado = df_nuevos

        guardados = 0
        if len(df_filtrado) > 0:
            df_filtrado.to_sql("matches", conn, if_exists="append", index=False)
            guardados = len(df_filtrado)
        conn.close()

        print(f"Partidos nuevos: {guardados} / Total encontrados: {len(lista)}")
        return guardados

    except Exception as e:
        print(f"Error extraccion: {e}")
        return 0

# ── PREDICCIONES DE ALTA CONFIANZA ────────────────────────
def generar_predicciones_destacadas():
    try:
        with open(MOD_PATH, 'rb') as f:
            paquete = pickle.load(f)

        modelos      = paquete['modelos']
        le_res       = paquete['le_res']
        le_sets      = paquete['le_sets']
        feature_cols = paquete['feature_cols']

        conn     = sqlite3.connect(DB_PATH)
        features = pd.read_sql("SELECT * FROM player_features", conn)
        h2h_df   = pd.read_sql("SELECT * FROM head_to_head", conn)
        df_hist  = pd.read_sql("SELECT * FROM matches_finalizados", conn)

        FECHA_HOY = datetime.now().strftime('%Y-%m-%d')
        try:
            pendientes = pd.read_sql(
                f"SELECT * FROM matches WHERE DATE(startTimestamp) = '{FECHA_HOY}' "
                f"AND status_type_finished = 0 LIMIT 20", conn
            )
        except:
            pendientes = pd.DataFrame()
        conn.close()

        if pendientes.empty:
            print("Sin partidos pendientes para predecir hoy")
            return []

        predicciones_altas = []

        for _, partido in pendientes.iterrows():
            try:
                local     = str(partido.get('homeTeam_name', partido.get('jugador_local', '')))
                visitante = str(partido.get('awayTeam_name', partido.get('jugador_visitante', '')))
                hora      = 15

                if not local or not visitante:
                    continue

                # Buscar stats
                fl = features[features['jugador'] == local]
                fv = features[features['jugador'] == visitante]
                if fl.empty or fv.empty:
                    continue

                fl = fl.iloc[0]
                fv = fv.iloc[0]

                mask_h2h = (
                    ((h2h_df['jugador_a'] == local) & (h2h_df['jugador_b'] == visitante)) |
                    ((h2h_df['jugador_a'] == visitante) & (h2h_df['jugador_b'] == local))
                )
                h2h_row   = h2h_df[mask_h2h]
                h2h_score = 0.5
                h2h_n     = 0
                if not h2h_row.empty:
                    f     = h2h_row.iloc[0]
                    total = f['partidos']
                    h2h_score = round(f['victorias_a'] / total, 3) if f['jugador_a'] == local else round(f['victorias_b'] / total, 3)
                    h2h_n = total

                df_hist_copy = df_hist.copy()
                df_hist_copy['hora_col'] = pd.to_datetime(df_hist_copy['fecha'], errors='coerce').dt.hour

                X = pd.DataFrame([{
                    'local_win_rate'   : fl['win_rate'],
                    'local_sets_ratio' : fl['sets_ratio'],
                    'local_racha'      : fl['victorias_ultimos_5'],
                    'local_partidos'   : fl['partidos'],
                    'visit_win_rate'   : fv['win_rate'],
                    'visit_sets_ratio' : fv['sets_ratio'],
                    'visit_racha'      : fv['victorias_ultimos_5'],
                    'visit_partidos'   : fv['partidos'],
                    'diff_win_rate'    : fl['win_rate'] - fv['win_rate'],
                    'diff_sets_ratio'  : fl['sets_ratio'] - fv['sets_ratio'],
                    'diff_racha'       : fl['victorias_ultimos_5'] - fv['victorias_ultimos_5'],
                    'h2h_local'        : h2h_score,
                    'h2h_partidos'     : h2h_n,
                    'hora'             : hora,
                    'wr_local_hora'    : 0.5,
                    'wr_visit_hora'    : 0.5,
                    'diff_wr_hora'     : 0.0,
                }])

                prob = modelos['ganador_partido'].predict_proba(X)[0]
                confianza = round(max(prob) * 100, 1)
                ganador   = local if prob[1] > prob[0] else visitante

                if confianza >= 75:
                    predicciones_altas.append({
                        'local'     : local,
                        'visitante' : visitante,
                        'ganador'   : ganador,
                        'confianza' : confianza,
                    })

            except Exception as e:
                continue

        return predicciones_altas

    except Exception as e:
        print(f"Error predicciones: {e}")
        return []

# ── EJECUCION PRINCIPAL ───────────────────────────────────
def ejecutar():
    print(f"\n{'='*50}")
    print(f"SCHEDULER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # 1. Extraer partidos
    nuevos = extraer_partidos()

    # 2. Notificar si hay partidos nuevos
    if nuevos > 0:
        enviar_telegram(
            f"🏓 <b>Czech Liga Pro</b>\n"
            f"✅ {nuevos} partidos nuevos guardados\n"
            f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        )

    # 3. Predicciones de alta confianza
    predicciones = generar_predicciones_destacadas()
    for p in predicciones:
        enviar_telegram(
            f"🎯 <b>PREDICCION ALTA CONFIANZA</b>\n"
            f"⚔️ {p['local']} vs {p['visitante']}\n"
            f"🏆 Ganador: <b>{p['ganador']}</b>\n"
            f"📊 Confianza: <b>{p['confianza']}%</b>"
        )

    print(f"Scheduler completado. Nuevos: {nuevos}, Predicciones altas: {len(predicciones)}")

if __name__ == '__main__':
    ejecutar()
