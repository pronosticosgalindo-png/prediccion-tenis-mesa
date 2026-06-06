import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("MODELO AVANZADO - CZECH LIGA PRO")
print("="*55)

# ── 1. CARGAR DATOS ────────────────────────────────────────
conexion = sqlite3.connect("base_ping_pong.sqlite")
df       = pd.read_sql("SELECT * FROM matches_finalizados", conexion)
features = pd.read_sql("SELECT * FROM player_features", conexion)
h2h      = pd.read_sql("SELECT * FROM head_to_head", conexion)
conexion.close()

df = df[df['total_sets'] > 0].copy()
df['fecha'] = pd.to_datetime(df['fecha'])
df['hora']  = df['fecha'].dt.hour

# Calcular puntos por set
for i in range(1, 6):
    col_l = f'puntos_local_s{i}'
    col_v = f'puntos_visitante_s{i}'
    if col_l in df.columns and col_v in df.columns:
        df[f'puntos_set{i}'] = (
            pd.to_numeric(df[col_l], errors='coerce') +
            pd.to_numeric(df[col_v], errors='coerce')
        )

df['puntos_totales'] = df[[f'puntos_set{i}' for i in range(1,6)
                           if f'puntos_set{i}' in df.columns]].sum(axis=1, skipna=True)

print(f"Partidos cargados: {len(df)}")

# ── 2. FUNCIONES DE SOPORTE ───────────────────────────────
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

def get_win_rate_horario(jugador, hora, df_hist):
    franja = 'manana' if hora < 12 else 'tarde' if hora < 18 else 'noche'
    if franja == 'manana':
        mask_hora = df_hist['hora'] < 12
    elif franja == 'tarde':
        mask_hora = (df_hist['hora'] >= 12) & (df_hist['hora'] < 18)
    else:
        mask_hora = df_hist['hora'] >= 18

    mask_jug = (
        (df_hist['jugador_local'] == jugador) |
        (df_hist['jugador_visitante'] == jugador)
    )
    df_jug = df_hist[mask_jug & mask_hora]

    if len(df_jug) == 0:
        return 0.5

    victorias = (df_jug['ganador'] == jugador).sum()
    return round(victorias / len(df_jug), 3)

# ── 3. CONSTRUIR DATASET ───────────────────────────────────
print("\nConstruyendo dataset avanzado...")
rows = []

for _, partido in df.iterrows():
    local     = partido['jugador_local']
    visitante = partido['jugador_visitante']
    ganador   = partido['ganador']
    hora      = partido['hora'] if not pd.isna(partido['hora']) else 12

    if ganador not in [local, visitante]:
        continue
    if pd.isna(partido['total_sets']) or partido['total_sets'] == 0:
        continue

    fl             = get_stats(local, features)
    fv             = get_stats(visitante, features)
    h2h_score, h2h_n = get_h2h_score(local, visitante, h2h)

    # Win rate por horario
    wr_local_hora = get_win_rate_horario(local, hora, df)
    wr_visit_hora = get_win_rate_horario(visitante, hora, df)

    # Ganador set 1
    pts_l_s1 = pd.to_numeric(partido.get('puntos_local_s1', 0), errors='coerce')
    pts_v_s1 = pd.to_numeric(partido.get('puntos_visitante_s1', 0), errors='coerce')
    if pd.isna(pts_l_s1) or pd.isna(pts_v_s1):
        gano_set1_local = 0.5
    else:
        gano_set1_local = 1 if pts_l_s1 > pts_v_s1 else 0

    # Resultado exacto
    sets_l = int(partido['sets_local']) if not pd.isna(partido['sets_local']) else 0
    sets_v = int(partido['sets_visitante']) if not pd.isna(partido['sets_visitante']) else 0
    resultado = f"{sets_l}-{sets_v}"

    rows.append({
        # Features base
        'local_win_rate'      : fl['win_rate'],
        'local_sets_ratio'    : fl['sets_ratio'],
        'local_racha'         : fl['victorias_ultimos_5'],
        'local_partidos'      : fl['partidos'],
        'visit_win_rate'      : fv['win_rate'],
        'visit_sets_ratio'    : fv['sets_ratio'],
        'visit_racha'         : fv['victorias_ultimos_5'],
        'visit_partidos'      : fv['partidos'],
        'diff_win_rate'       : fl['win_rate']   - fv['win_rate'],
        'diff_sets_ratio'     : fl['sets_ratio'] - fv['sets_ratio'],
        'diff_racha'          : fl['victorias_ultimos_5'] - fv['victorias_ultimos_5'],
        'h2h_local'           : h2h_score,
        'h2h_partidos'        : h2h_n,
        # Features avanzadas
        'hora'                : hora,
        'wr_local_hora'       : wr_local_hora,
        'wr_visit_hora'       : wr_visit_hora,
        'diff_wr_hora'        : wr_local_hora - wr_visit_hora,
        # Targets
        'gano_local'          : 1 if ganador == local else 0,
        'total_sets'          : int(partido['total_sets']),
        'resultado_sets'      : resultado,
        'gano_set1_local'     : gano_set1_local,
        'puntos_totales'      : partido['puntos_totales'],
    })

dataset = pd.DataFrame(rows)
print(f"Dataset construido: {len(dataset)} partidos")

feature_cols = [
    'local_win_rate', 'local_sets_ratio', 'local_racha', 'local_partidos',
    'visit_win_rate', 'visit_sets_ratio', 'visit_racha', 'visit_partidos',
    'diff_win_rate',  'diff_sets_ratio',  'diff_racha',
    'h2h_local',      'h2h_partidos',
    'hora',           'wr_local_hora',    'wr_visit_hora', 'diff_wr_hora',
]

X = dataset[feature_cols]

print(f"\n{'='*55}")
print("ENTRENANDO MODELOS AVANZADOS")
print("="*55)

cv5  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv5r = KFold(n_splits=5, shuffle=True, random_state=42)
modelos_guardados = {}

# ── 4. MODELO — GANADOR PARTIDO ───────────────────────────
print("\nModelo 1 - Ganador del partido...")
y = dataset['gano_local']
m = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                max_depth=4, random_state=42)
scores = cross_val_score(m, X, y, cv=cv5, scoring='accuracy')
m.fit(X, y)
modelos_guardados['ganador_partido'] = m
print(f"  Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

# ── 5. MODELO — GANADOR SET 1 ─────────────────────────────
print("\nModelo 2 - Ganador del Set 1...")
df_set1 = dataset[dataset['gano_set1_local'].isin([0, 1])].copy()
y_set1  = df_set1['gano_set1_local'].astype(int)
X_set1  = df_set1[feature_cols]
m2 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                 max_depth=4, random_state=42)
scores2 = cross_val_score(m2, X_set1, y_set1, cv=cv5, scoring='accuracy')
m2.fit(X_set1, y_set1)
modelos_guardados['ganador_set1'] = m2
print(f"  Accuracy: {scores2.mean():.3f} +/- {scores2.std():.3f}")

# ── 6. MODELO — RESULTADO EXACTO ──────────────────────────
print("\nModelo 3 - Resultado exacto de sets...")
resultados_validos = ['3-0', '3-1', '3-2', '0-3', '1-3', '2-3']
df_res = dataset[dataset['resultado_sets'].isin(resultados_validos)].copy()
le_res = LabelEncoder()
y_res  = le_res.fit_transform(df_res['resultado_sets'])
X_res  = df_res[feature_cols]
m3 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                 max_depth=4, random_state=42)
scores3 = cross_val_score(m3, X_res, y_res, cv=cv5, scoring='accuracy')
m3.fit(X_res, y_res)
modelos_guardados['resultado_exacto'] = m3
print(f"  Accuracy: {scores3.mean():.3f} +/- {scores3.std():.3f}")
print(f"  Clases  : {list(le_res.classes_)}")

# ── 7. MODELO — TOTAL SETS ────────────────────────────────
print("\nModelo 4 - Total de sets jugados...")
def cat_sets(s):
    if s <= 3:   return '3_sets'
    elif s == 4: return '4_sets'
    else:        return '5_sets'

y_sets    = dataset['total_sets'].apply(cat_sets)
le_sets   = LabelEncoder()
y_sets_enc = le_sets.fit_transform(y_sets)
m4 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                 max_depth=4, random_state=42)
scores4 = cross_val_score(m4, X, y_sets_enc, cv=cv5, scoring='accuracy')
m4.fit(X, y_sets_enc)
modelos_guardados['total_sets'] = m4
print(f"  Accuracy: {scores4.mean():.3f} +/- {scores4.std():.3f}")

# ── 8. MODELO — PUNTOS TOTALES ────────────────────────────
print("\nModelo 5 - Puntos totales del partido...")
df_pts = dataset[dataset['puntos_totales'] > 0].copy()
y_pts  = df_pts['puntos_totales']
X_pts  = df_pts[feature_cols]
m5 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                max_depth=4, random_state=42)
scores5 = cross_val_score(m5, X_pts, y_pts, cv=cv5r,
                           scoring='neg_mean_absolute_error')
m5.fit(X_pts, y_pts)
modelos_guardados['puntos_totales'] = m5
print(f"  Error medio: +/- {-scores5.mean():.1f} puntos")

# ── 9. MODELO — RENDIMIENTO POR HORARIO ───────────────────
print("\nModelo 6 - Rendimiento por horario...")
print(f"  Distribucion de partidos por franja:")
df['franja'] = df['hora'].apply(
    lambda h: 'manana' if h < 12 else 'tarde' if h < 18 else 'noche'
)
dist_hora = df['franja'].value_counts()
for franja, cnt in dist_hora.items():
    print(f"    {franja}: {cnt} partidos ({cnt/len(df)*100:.1f}%)")

# ── 10. GUARDAR TODO ───────────────────────────────────────
with open('analysis/modelo_avanzado.pkl', 'wb') as f:
    pickle.dump({
        'modelos'      : modelos_guardados,
        'le_res'       : le_res,
        'le_sets'      : le_sets,
        'feature_cols' : feature_cols,
        'accuracies'   : {
            'ganador_partido' : scores.mean(),
            'ganador_set1'    : scores2.mean(),
            'resultado_exacto': scores3.mean(),
            'total_sets'      : scores4.mean(),
            'puntos_totales'  : -scores5.mean(),
        }
    }, f)

# ── 11. REPORTE FINAL ──────────────────────────────────────
print(f"\n{'='*55}")
print("REPORTE FINAL - MODELOS AVANZADOS")
print("="*55)
print(f"  Modelo 1 - Ganador partido   : {scores.mean():.3f} accuracy")
print(f"  Modelo 2 - Ganador Set 1     : {scores2.mean():.3f} accuracy")
print(f"  Modelo 3 - Resultado exacto  : {scores3.mean():.3f} accuracy")
print(f"  Modelo 4 - Total sets        : {scores4.mean():.3f} accuracy")
print(f"  Modelo 5 - Puntos totales    : +/- {-scores5.mean():.1f} puntos")
print(f"  Modelo 6 - Horario           : incluido en features")
print(f"  Guardado en                  : analysis/modelo_avanzado.pkl")
print("="*55)
print("\nadvanced_model.py ejecutado correctamente")