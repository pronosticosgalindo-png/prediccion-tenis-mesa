import os
import pandas as pd
import requests # <-- ¡Nueva librería necesaria!

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# 1. RUTAS BASE
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

players_path = os.path.join(DATA_DIR, "players.csv")
matches_path = os.path.join(DATA_DIR, "matches.csv")

print("BASE_DIR:", BASE_DIR)
print("Contenido de data:", os.listdir(DATA_DIR))

# ==============================
# 1.5. EXTRACCIÓN SIGILOSA (API)
# ==============================

def actualizar_partidos():
    print("\nIniciando extracción sigilosa desde Sofascore...")
    url_api = "AQUI_PEGA_LA_URL_DEL_JSON_DE_SOFASCORE"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    respuesta = requests.get(url_api, headers=headers)
    if respuesta.status_code == 200:
        lista_partidos = respuesta.json().get('events', [])
        df_nuevos = pd.json_normalize(lista_partidos)
        
        # IMPORTANTE: Aquí deberás filtrar y renombrar las columnas de df_nuevos 
        # para que se llamen exactamente igual que las de tu matches.csv
        # Ejemplo rápido: df_nuevos = df_nuevos[['id', 'homeScore', ...]]
        
        # Añadimos los datos nuevos al CSV existente (modo 'a' de append)
        # Esto reemplaza el ingreso manual en Excel
        df_nuevos.to_csv(matches_path, mode='a', header=False, index=False)
        print(f"¡Éxito! Se añadieron {len(df_nuevos)} partidos a matches.csv")
    else:
        print(f"Error en la API: Código {respuesta.status_code}")

# --- EL INTERRUPTOR MÁGICO ---
# Cambia esto a True solo cuando quieras descargar datos nuevos.
# Déjalo en False cuando estés ajustando el modelo.
DESCARGAR_NUEVOS_DATOS = False 

if DESCARGAR_NUEVOS_DATOS:
    actualizar_partidos()

# ==============================
# 2. CARGA DE ARCHIVOS
# ==============================

print("\nLeyendo PLAYERS desde:", players_path)
players = pd.read_csv(players_path)
print(players.head())

print("\nLeyendo MATCHES desde:", matches_path)
matches = pd.read_csv(matches_path)
print(matches.head())

# ==============================
# 3. VALIDACIÓN DE COLUMNAS
# ==============================

required_matches_columns = [
    "match_id",
    "player_id",
    "opponent_id",
    "sets_won",
    "sets_lost",
    "points_won",
    "points_lost"
]

missing = [c for c in required_matches_columns if c not in matches.columns]
if missing:
    raise ValueError(f"Faltan columnas en matches.csv: {missing}")

required_players_columns = ["player_id", "player_name"]
missing_players = [c for c in required_players_columns if c not in players.columns]
if missing_players:
    raise ValueError(f"Faltan columnas en players.csv: {missing_players}")

# ==============================
# 4. CONEXIÓN MATCHES ↔ PLAYERS
# ==============================

df = matches.merge(
    players,
    on="player_id",
    how="left"
)

df = df.merge(
    players,
    left_on="opponent_id",
    right_on="player_id",
    how="left",
    suffixes=("", "_opponent")
)

df = df.rename(columns={
    "player_name": "player_name",
    "player_name_opponent": "opponent_name"
})

print("\nDatos conectados:")
print(df[[
    "match_id",
    "player_name",
    "opponent_name",
    "sets_won",
    "sets_lost"
]].head())

# ==============================
# 5. TARGET (VARIABLE A PREDECIR)
# ==============================

df["win"] = (df["sets_won"] > df["sets_lost"]).astype(int)

# ==============================
# 6. FEATURE ENGINEERING
# ==============================

df["sets_diff"] = df["sets_won"] - df["sets_lost"]
df["points_diff"] = df["points_won"] - df["points_lost"]

features = [
    "sets_diff",
    "points_diff",
    "points_won",
    "points_lost"
]

df = df.dropna(subset=features + ["win"])

print("\nFilas después de limpieza:", len(df))

# ==============================
# 7. DEFINICIÓN DE X e y
# ==============================

X = df[features]
y = df["win"]

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# ==============================
# 8. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# ==============================
# 9. MODELO
# ==============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# 10. EVALUACIÓN
# ==============================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy del modelo:", round(accuracy, 4))
print("✅ explore.py ejecutado correctamente")