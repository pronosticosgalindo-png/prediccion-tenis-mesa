import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from curl_cffi import requests

# ── CONFIGURACION ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'base_ping_pong.sqlite')
MOD_PATH = os.path.join(BASE_DIR, 'analysis', 'modelo_avanzado.pkl')

TORNEO_ID = '19039'

cookies = {
    '_cc_id': '665c530e0f6aa71a8a4cb0637145eaa7',
    '_ga': 'GA1.1.1956304625.1762201631',
    'AD_VERGE_SESSION_COOKIE_V1': '8f5072f4-2b49-4f5c-a190-6e99e705eb06',
    '_adv_uid': 'c97e1656-b739-4520-ad16-137ebb86720a',
    'hb_insticator_uid': '847eaa92-6071-4fbc-bd86-227c62f6fd6d',
    '_gcl_au': '1.1.835639633.1780018185',
    '_adv_sid': '7799cd2d-4229-48c2-8496-4616a6a962cd',
    'ssp_test': 'control',
    '__gads': 'ID=86ba3bf017fab095:T=1762201627:RT=1781314847:S=ALNI_MYXKVhFZ-tm3js_y8AGbLqMYgLU_A',
    '__gpi': 'UID=000012a59574e3c7:T=1762201627:RT=1781314847:S=ALNI_MYIUpKlhX_9Lvn6I8c9rYA1mKByZQ',
    '__eoi': 'ID=9f47ba8481905a21:T=1780018205:RT=1781314847:S=AA-AfjZK16GEX47BwSM7nZyc3Mj9',
    'cto_bundle': '_-c0x19YUmFhbXozMUdhcE01QUt3Q3FCbU8lMkIyWXNnSDlFbFRjbXREOUVwa25lVjNyQmlqM1FyVVdEcDJ1YjdnZkc2MUdZRnZMR3BGbEhGYmFzbWxvWVZoMVFiM2ZIT255MFhSJTJGeXVVMjRaV1JwQ1lrblRvNnJZV0JPb3U5TTRsWHFhd0xLbFclMkZKbmJXMGsyTWQ0VUlqbFROU2clM0QlM0Q',
    'cto_bidid': '3LAkXV9pVWljNUhzZEZRQkdiclVtTUQ2bSUyQlBYZnlTb2p5WW8lMkJ5YlglMkZEbDdDbWpNNCUyRlRXcCUyRnJUVHB6c0ZSc0NJeE1uNzFkNkd5JTJGOEFsdzE3Y2E5amt4YzdhdlBrV09YSDFqSGYlMkY1UnJVVEMzV1VZJTNE',
    'FCCDCF': '%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%22398b09f5-8b5b-4d0d-8a97-49e5c52d93ce%5C%22%2C%5B1762201626%2C136000000%5D%5D%22%5D%5D%5D',
    'FCNEC': '%5B%5B%22AKsRol9BpLUSC47Z1hzd4T3DKtzbfCXIt78Ovnh3X6XVhfT0hLmBiX7zrYSp4daOfAh0f0Os3uUbJSrcwxbSCyAzvH2kZ5V1iVJ7syiAKAzJONqJa7DH8qwDKffLGnxX7W0CZV66mSVP6sq_aC0giVMvOEr97E3VKA%3D%3D%22%5D%5D',
    '_ga_HNQ9P9MGZR': 'GS2.1.s1781314860$o26$g1$t1781315052$j60$l0$h0',
}

headers = {
    'accept': '*/*',
    'accept-language': 'es-ES,es;q=0.9',
    'cache-control': 'max-age=0',
    'referer': 'https://www.sofascore.com/es/table-tennis',
    'sec-ch-ua': '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36',
    'x-requested-with': 'd79e08',
}

# ── EXTRACTOR ─────────────────────────────────────────────
def extraer_partidos():
    FECHA_HOY = datetime.now().strftime('%Y-%m-%d')
    print(f"Extrayendo partidos: {FECHA_HOY}")
    try:
        response = requests.get(
            f'https://www.sofascore.com/api/v1/unique-tournament/{TORNEO_ID}/scheduled-events/{FECHA_HOY}',
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
            df_exist    = pd.read_sql("SELECT id FROM matches", conn)
            ids_exist   = set(df_exist['id'].astype(str).tolist())
            df_filtrado = df_nuevos[~df_nuevos['id'].astype(str).isin(ids_exist)]
        except:
            df_filtrado = df_nuevos

        guardados = 0
        if len(df_filtrado) > 0:
            df_filtrado.to_sql("matches", conn, if_exists="append", index=False)
            guardados = len(df_filtrado)
        conn.close()
        print(f"Partidos nuevos: {guardados} / Total: {len(lista)}")
        return guardados
    except Exception as e:
        print(f"Error extraccion: {e}")
        return 0

# ── PREDICCIONES DESTACADAS ───────────────────────────────
def generar_predicciones_destacadas():
    try:
        with open(MOD_PATH, 'rb') as f:
            paquete = pickle.load(f)
        modelos = paquete['modelos']

        conn     = sqlite3.connect(DB_PATH)
        features = pd.read_sql("SELECT * FROM player_features", conn)
        h2h_df   = pd.read_sql("SELECT * FROM head_to_head", conn)
        FECHA_HOY = datetime.now().strftime('%Y-%m-%d')

        try:
            pendientes = pd.read_sql(
                "SELECT * FROM matches WHERE DATE(startTimestamp, 'unixepoch') = ? LIMIT 20",
                conn, params=(FECHA_HOY,)
            )
        except:
            pendientes = pd.DataFrame()
        conn.close()

        if pendientes.empty:
            return []

        predicciones_altas = []
        for _, partido in pendientes.iterrows():
            try:
                local     = str(partido.get('homeTeam_name', ''))
                visitante = str(partido.get('awayTeam_name', ''))
                if not local or not visitante or local == 'nan':
                    continue

                fl = features[features['jugador'] == local]
                fv = features[features['jugador'] == visitante]
                if fl.empty or fv.empty:
                    continue

                fl = fl.iloc[0]
                fv = fv.iloc[0]

                mask = (
                    ((h2h_df['jugador_a'] == local) & (h2h_df['jugador_b'] == visitante)) |
                    ((h2h_df['jugador_a'] == visitante) & (h2h_df['jugador_b'] == local))
                )
                h2h_row   = h2h_df[mask]
                h2h_score = 0.5
                h2h_n     = 0
                if not h2h_row.empty:
                    f = h2h_row.iloc[0]
                    h2h_score = round(f['victorias_a'] / f['partidos'], 3) if f['jugador_a'] == local else round(f['victorias_b'] / f['partidos'], 3)
                    h2h_n = f['partidos']

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
                    'hora'             : 15,
                    'wr_local_hora'    : 0.5,
                    'wr_visit_hora'    : 0.5,
                    'diff_wr_hora'     : 0.0,
                }])

                prob      = modelos['ganador_partido'].predict_proba(X)[0]
                confianza = round(max(prob) * 100, 1)
                ganador   = local if prob[1] > prob[0] else visitante

                if confianza >= 75:
                    predicciones_altas.append({
                        'local': local, 'visitante': visitante,
                        'ganador': ganador, 'confianza': confianza,
                    })
            except:
                continue

        return predicciones_altas
    except Exception as e:
        print(f"Error predicciones: {e}")
        return []

# ── GUARDAR ALERTAS EN BASE DE DATOS ──────────────────────
def guardar_alertas(nuevos, predicciones):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS alertas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tipo TEXT,
                mensaje TEXT,
                fecha TEXT,
                leida INTEGER DEFAULT 0
            )
        ''')

        if nuevos > 0:
            conn.execute(
                "INSERT INTO alertas (tipo, mensaje, fecha) VALUES (?, ?, ?)",
                ('nuevos_partidos',
                 f'{nuevos} partidos nuevos guardados hoy',
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )

        for p in predicciones:
            conn.execute(
                "INSERT INTO alertas (tipo, mensaje, fecha) VALUES (?, ?, ?)",
                ('prediccion_alta',
                 f"{p['local']} vs {p['visitante']} — Ganador: {p['ganador']} ({p['confianza']}%)",
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )

        conn.commit()
        conn.close()
        print(f"Alertas guardadas en base de datos")
    except Exception as e:
        print(f"Error guardando alertas: {e}")

# ── EJECUCION PRINCIPAL ───────────────────────────────────
def ejecutar():
    print(f"\n{'='*50}")
    print(f"SCHEDULER — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    nuevos       = extraer_partidos()
    predicciones = generar_predicciones_destacadas()
    guardar_alertas(nuevos, predicciones)

    print(f"\nScheduler completado.")
    print(f"  Partidos nuevos   : {nuevos}")
    print(f"  Predicciones altas: {len(predicciones)}")

if __name__ == '__main__':
    ejecutar()
