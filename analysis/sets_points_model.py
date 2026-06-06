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
print("MODELO DE SETS Y PUNTOS - CZECH LIGA PRO")
print("="*55)

# ── 1. CARGAR DATOS ────────────────────────────────────────
conexion = sqlite3.connect("base_ping_pong.sqlite")
df       = pd.read_sql("SELECT * FROM matches_finalizados", conexion)
features = pd.read_sql("SELECT * FROM player_features", conexion)
h2h      = pd.read_sql("SELECT * FROM head_to_head", conexion)
conexion.close()

# Filtrar solo partidos con datos completos
df = df[df['total_sets'] > 0].copy()
df['fecha'] = pd.to_datetime(df['fecha'])

print(f"Partidos con datos completos: {len(df)}")

# ── 2. CALCULAR PUNTOS TOTALES POR SET ─────────────────────
cols_sets = [
    ('puntos_local_s1', 'puntos_visitante_s1'),
    ('puntos_local_s2', 'puntos_visitante_s2'),
    ('puntos_local_s3', 'puntos_visitante_s3'),
    ('puntos_local_s4', 'puntos_visitante_s4'),
    ('puntos_local_s5', 'puntos_visitante_s5'),
]

for i, (col_l, col_v) in enumerate(cols_sets, 1):
    if col_l in df.columns and col_v in df.columns:
        df[f'puntos_set{i}'] = (
            pd.to_numeric(df[col_l], errors='coerce') +
            pd.to_numeric(df[col_v], errors='coerce')
        )

# Puntos totales del partido
puntos_cols = [f'puntos_set{i}' for i in range(1, 6) if f'puntos_set{i}' in df.columns]
df['puntos_totales'] = df[puntos_cols].sum(axis=1, skipna=True)

# Promedio de puntos por set
df['promedio_puntos_set'] = (df['puntos_totales'] / df['total_sets']).round(1)

print(f"\nEstadisticas generales:")
print(f"  Sets promedio por partido : {df['total_sets'].mean():.2f}")
print(f"  Puntos promedio por set   : {df['promedio_puntos_set'].mean():.2f}")
print(f"  Puntos totales promedio   : {df['puntos_totales'].mean():.2f}")
print(f"  Min puntos totales        : {df['puntos_totales'].min():.0f}")
print(f"  Max puntos totales        : {df['puntos_totales'].max():.0f}")

# ── 3. DISTRIBUCION DE SETS ────────────────────────────────
print(f"\nDistribucion de sets jugados:")
dist_sets = df['total_sets'].value_counts().sort_index()
for sets, count in dist_sets.items():
    pct = count / len(df) * 100
    print(f"  {int(sets)} sets: {count:>5} partidos ({pct:.1f}%)")

# ── 4. FUNCIONES DE SOPORTE ───────────────────────────────
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
        return 0.5, 0
    f = fila.iloc[0]
    total = f['partidos']
    if f['jugador_a'] == jugador_a:
        return round(f['victorias_a'] / total, 3), total
    else:
        return round(f['victorias_b'] / total, 3), total

# ── 5. CONSTRUIR DATASET ───────────────────────────────────
print(f"\nConstruyendo dataset...")
rows = []

for _, partido in df.iterrows():
    local     = partido['jugador_local']
    visitante = partido['jugador_visitante']

    if pd.isna(partido['total_sets']) or partido['total_sets'] == 0:
        continue
    if pd.isna(partido['puntos_totales']) or partido['puntos_totales'] == 0:
        continue

    fl             = get_features_jugador(local, features)
    fv             = get_features_jugador(visitante, features)
    h2h_score, h2h_partidos = get_h2h(local, visitante, h2h)

    rows.append({
        # Features jugadores
        'local_win_rate'        : fl['win_rate'],
        'local_sets_ratio'      : fl['sets_ratio'],
        'local_racha'           : fl['victorias_ultimos_5'],
        'local_partidos'        : fl['partidos'],
        'visit_win_rate'        : fv['win_rate'],
        'visit_sets_ratio'      : fv['sets_ratio'],
        'visit_racha'           : fv['victorias_ultimos_5'],
        'visit_partidos'        : fv['partidos'],
        # Diferencias
        'diff_win_rate'         : fl['win_rate']   - fv['win_rate'],
        'diff_sets_ratio'       : fl['sets_ratio'] - fv['sets_ratio'],
        'diff_racha'            : fl['victorias_ultimos_5'] - fv['victorias_ultimos_5'],
        # H2H
        'h2h_local'             : h2h_score,
        'h2h_partidos'          : h2h_partidos,
        # Targets
        'total_sets'            : int(partido['total_sets']),
        'puntos_totales'        : partido['puntos_totales'],
        'promedio_puntos_set'   : partido['promedio_puntos_set'],
    })

dataset = pd.DataFrame(rows)
print(f"Dataset construido: {len(dataset)} partidos")

feature_cols = [
    'local_win_rate', 'local_sets_ratio', 'local_racha', 'local_partidos',
    'visit_win_rate', 'visit_sets_ratio', 'visit_racha', 'visit_partidos',
    'diff_win_rate',  'diff_sets_ratio',  'diff_racha',
    'h2h_local',      'h2h_partidos',
]

X = dataset[feature_cols]

# ── 6. MODELO A — CLASIFICACION DE SETS ───────────────────
print(f"\n{'='*55}")
print("MODELO A - PREDICCION DE SETS JUGADOS")
print("="*55)

# Categorias de sets
def categorizar_sets(s):
    if s <= 3:   return '3_sets'
    elif s == 4: return '4_sets'
    else:        return '5_sets'

y_sets = dataset['total_sets'].apply(categorizar_sets)

dist = y_sets.value_counts()
for cat, cnt in dist.items():
    print(f"  {cat}: {cnt} partidos ({cnt/len(y_sets)*100:.1f}%)")

le_sets = LabelEncoder()
y_sets_enc = le_sets.fit_transform(y_sets)

modelo_sets = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, random_state=42
)

cv_sets = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_sets = cross_val_score(modelo_sets, X, y_sets_enc, cv=cv_sets, scoring='accuracy')
modelo_sets.fit(X, y_sets_enc)

print(f"\n  Accuracy sets: {scores_sets.mean():.3f} +/- {scores_sets.std():.3f}")

# ── 7. MODELO B — REGRESION PUNTOS TOTALES ─────────────────
print(f"\n{'='*55}")
print("MODELO B - PREDICCION PUNTOS TOTALES")
print("="*55)

y_puntos = dataset['puntos_totales']

modelo_puntos = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, random_state=42
)

cv_puntos = KFold(n_splits=5, shuffle=True, random_state=42)
scores_puntos = cross_val_score(
    modelo_puntos, X, y_puntos, cv=cv_puntos, scoring='neg_mean_absolute_error'
)
modelo_puntos.fit(X, y_puntos)

mae = -scores_puntos.mean()
print(f"  Error medio puntos totales: +/- {mae:.1f} puntos")

# ── 8. MODELO C — REGRESION PUNTOS POR SET ─────────────────
print(f"\n{'='*55}")
print("MODELO C - PREDICCION PUNTOS PROMEDIO POR SET")
print("="*55)

y_prom = dataset['promedio_puntos_set']

modelo_prom = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, random_state=42
)

scores_prom = cross_val_score(
    modelo_prom, X, y_prom, cv=cv_puntos, scoring='neg_mean_absolute_error'
)
modelo_prom.fit(X, y_prom)

mae_prom = -scores_prom.mean()
print(f"  Error medio puntos por set: +/- {mae_prom:.1f} puntos")

# ── 9. GUARDAR MODELOS ─────────────────────────────────────
with open('analysis/modelo_sets_puntos.pkl', 'wb') as f:
    pickle.dump({
        'modelo_sets'    : modelo_sets,
        'modelo_puntos'  : modelo_puntos,
        'modelo_prom'    : modelo_prom,
        'le_sets'        : le_sets,
        'feature_cols'   : feature_cols,
        'accuracy_sets'  : scores_sets.mean(),
        'mae_puntos'     : mae,
        'mae_prom'       : mae_prom,
        'stats': {
            'sets_promedio'   : df['total_sets'].mean(),
            'puntos_promedio' : df['puntos_totales'].mean(),
            'puntos_min'      : df['puntos_totales'].min(),
            'puntos_max'      : df['puntos_totales'].max(),
            'prom_set'        : df['promedio_puntos_set'].mean(),
        }
    }, f)

# ── 10. REPORTE FINAL ──────────────────────────────────────
print(f"\n{'='*55}")
print("REPORTE FINAL")
print("="*55)
print(f"  Modelo A - Sets jugados    : {scores_sets.mean():.3f} accuracy")
print(f"  Modelo B - Puntos totales  : +/- {mae:.1f} puntos de error")
print(f"  Modelo C - Puntos por set  : +/- {mae_prom:.1f} puntos de error")
print(f"  Modelos guardados en       : analysis/modelo_sets_puntos.pkl")
print("="*55)
print("\nsets_points_model.py ejecutado correctamente")