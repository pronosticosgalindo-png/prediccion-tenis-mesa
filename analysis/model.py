import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Iniciando entrenamiento del modelo...")
print("="*55)

# ── 1. CARGAR DATOS ────────────────────────────────────────
conexion = sqlite3.connect("base_ping_pong.sqlite")
df       = pd.read_sql("SELECT * FROM matches_finalizados", conexion)
features = pd.read_sql("SELECT * FROM player_features", conexion)
h2h      = pd.read_sql("SELECT * FROM head_to_head", conexion)
conexion.close()

print(f"Partidos cargados        : {len(df)}")
print(f"Jugadores con features   : {len(features)}")
print(f"Cruces head-to-head      : {len(h2h)}")

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

# ── 3. CONSTRUIR DATASET ───────────────────────────────────
print("\nConstruyendo dataset de entrenamiento...")
rows = []
for _, partido in df.iterrows():
    local     = partido['jugador_local']
    visitante = partido['jugador_visitante']
    ganador   = partido['ganador']

    if ganador not in [local, visitante]:
        continue

    fl        = get_features_jugador(local, features)
    fv        = get_features_jugador(visitante, features)
    h2h_score = get_h2h(local, visitante, h2h)

    rows.append({
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
        'gano_local'       : 1 if ganador == local else 0,
    })

dataset = pd.DataFrame(rows)
print(f"Dataset construido       : {len(dataset)} partidos")
print(f"Victorias local          : {dataset['gano_local'].sum()}")
print(f"Victorias visitante      : {(dataset['gano_local'] == 0).sum()}")

# ── 4. FEATURES Y TARGET ───────────────────────────────────
feature_cols = [
    'local_win_rate', 'local_sets_ratio', 'local_racha', 'local_partidos',
    'visit_win_rate', 'visit_sets_ratio', 'visit_racha', 'visit_partidos',
    'diff_win_rate',  'diff_sets_ratio',  'diff_racha',  'h2h_local',
]

X = dataset[feature_cols]
y = dataset['gano_local']

# ── 5. COMPARAR MODELOS ────────────────────────────────────
print(f"\n{'='*55}")
print("COMPARACION DE MODELOS")
print("="*55)

modelos = {
    'Random Forest'       : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=200, random_state=42),
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}

for nombre, modelo in modelos.items():
    try:
        scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy')
        resultados[nombre] = scores.mean()
        print(f"  {nombre:<25} -> Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    except Exception as e:
        print(f"  {nombre:<25} -> Error: {e}")
        resultados[nombre] = 0.0

# ── 6. MEJOR MODELO ────────────────────────────────────────
mejor_nombre = max(resultados, key=resultados.get)
mejor_modelo = modelos[mejor_nombre]
mejor_modelo.fit(X, y)

print(f"\nMejor modelo: {mejor_nombre}")
print(f"Accuracy    : {resultados[mejor_nombre]:.3f}")

# ── 7. IMPORTANCIA DE FEATURES ────────────────────────────
if hasattr(mejor_modelo, 'feature_importances_'):
    importancias = pd.DataFrame({
        'feature'    : feature_cols,
        'importancia': mejor_modelo.feature_importances_
    }).sort_values('importancia', ascending=False)

    print(f"\nIMPORTANCIA DE FEATURES ({mejor_nombre}):")
    print(importancias.to_string(index=False))

# ── 8. GUARDAR MODELO ──────────────────────────────────────
with open('analysis/modelo_ping_pong.pkl', 'wb') as f:
    pickle.dump({
        'modelo'      : mejor_modelo,
        'feature_cols': feature_cols,
        'nombre'      : mejor_nombre,
        'accuracy'    : resultados[mejor_nombre],
    }, f)

# ── 9. REPORTE FINAL ───────────────────────────────────────
print(f"\n{'='*55}")
print("REPORTE DEL MODELO")
print("="*55)
print(f"  Modelo seleccionado  : {mejor_nombre}")
print(f"  Accuracy (CV)        : {resultados[mejor_nombre]:.3f}")
print(f"  Features utilizadas  : {len(feature_cols)}")
print(f"  Partidos entrenados  : {len(dataset)}")
print(f"  Modelo guardado en   : analysis/modelo_ping_pong.pkl")
print("="*55)
print("\nmodel.py ejecutado correctamente")