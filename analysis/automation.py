import sys
sys.stdout.reconfigure(encoding='utf-8')
import subprocess
import sqlite3
import pandas as pd
from datetime import datetime
import time
import os

print("="*55)
print("SISTEMA DE AUTOMATIZACION - CZECH LIGA PRO")
print("="*55)
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*55)

# ── 1. CONFIGURACION ───────────────────────────────────────
PYTHON  = sys.executable
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = f"{LOG_DIR}/automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

PIPELINE = [
    ("czech_extractor.py",    "Extrayendo datos Czech Liga Pro...",  True),
    ("processor.py",          "Procesando y limpiando datos...",     True),
    ("features.py",           "Calculando features...",              True),
    ("model.py",              "Entrenando modelo...",                True),
    ("advanced_model.py",     "Entrenando modelos avanzados...",     False),
    ("predictor.py",          "Generando predicciones...",           False),
    ("advanced_predictor.py", "Generando predicciones avanzadas...", False),
]

resultados_pipeline = []

# ── 2. FUNCION DE EJECUCION ────────────────────────────────
def ejecutar_script(script, descripcion, critico):
    print(f"\n{descripcion}")
    inicio = time.time()

    resultado = subprocess.run(
        [PYTHON, f"analysis/{script}"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    )

    duracion = round(time.time() - inicio, 2)
    exito    = resultado.returncode == 0

    if exito:
        print(f"  OK - Completado en {duracion}s")
    else:
        print(f"  ERROR en {script}:")
        print(f"  {resultado.stderr[-400:]}")

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*55}\n")
        f.write(f"Script  : {script}\n")
        f.write(f"Estado  : {'OK' if exito else 'ERROR'}\n")
        f.write(f"Duracion: {duracion}s\n")
        f.write(f"Output  :\n{resultado.stdout}\n")
        if resultado.stderr:
            f.write(f"Errors  :\n{resultado.stderr}\n")

    return {
        'script'  : script,
        'exito'   : exito,
        'duracion': duracion,
        'critico' : critico,
    }

# ── 3. EJECUTAR PIPELINE ───────────────────────────────────
print("\nEJECUTANDO PIPELINE...")
pipeline_detenido = False

for script, descripcion, critico in PIPELINE:
    r = ejecutar_script(script, descripcion, critico)
    resultados_pipeline.append(r)

    if not r['exito'] and critico:
        print(f"\nPipeline detenido - error critico en {script}")
        pipeline_detenido = True
        break

# ── 4. PREDICCIONES DEL DIA ────────────────────────────────
print(f"\n{'='*55}")
print("PREDICCIONES DEL DIA - CZECH LIGA PRO")
print("="*55)

try:
    conexion = sqlite3.connect("base_ping_pong.sqlite")
    pred     = pd.read_sql("SELECT * FROM predicciones", conexion)

    if len(pred) > 0:
        # Separar por nivel de confianza
        altas  = pred[pred['confianza'] >= 80].sort_values('confianza', ascending=False)
        medias = pred[(pred['confianza'] >= 60) & (pred['confianza'] < 80)]
        bajas  = pred[pred['confianza'] < 60]

        print(f"\n  Total predicciones : {len(pred)}")
        print(f"  Confianza ALTA     : {len(altas)}")
        print(f"  Confianza MEDIA    : {len(medias)}")
        print(f"  Confianza BAJA     : {len(bajas)}")

        if len(altas) > 0:
            print(f"\n  {'='*45}")
            print(f"  RECOMENDADAS PARA RUSHBET [ALTA]")
            print(f"  {'='*45}")
            for _, r in altas.iterrows():
                print(f"\n  {r.get('fecha', 'Sin fecha')}")
                print(f"  {r['local']} vs {r['visitante']}")
                print(f"  Ganador  : {r['prediccion']}")
                print(f"  Prob     : {r['local']} {r['prob_local']}% - {r['visitante']} {r['prob_visitante']}%")
                print(f"  Confianza: {r['confianza']}% [ALTA]")
                print(f"  {'-'*45}")

        if len(medias) > 0:
            print(f"\n  CONFIANZA MEDIA - Con precaucion")
            print(f"  {'-'*45}")
            for _, r in medias.iterrows():
                print(f"  {r['local']} vs {r['visitante']} -> {r['prediccion']} ({r['confianza']}%)")

        if len(bajas) > 0:
            print(f"\n  CONFIANZA BAJA - No recomendadas")
            print(f"  {'-'*45}")
            for _, r in bajas.iterrows():
                print(f"  {r['local']} vs {r['visitante']} -> {r['prediccion']} ({r['confianza']}%)")
    else:
        print("  Sin predicciones para hoy.")

except Exception as e:
    print(f"  Error leyendo predicciones: {e}")

# ── 5. ESTADO DE LA BASE DE DATOS ─────────────────────────
print(f"\n{'='*55}")
print("ESTADO DE LA BASE DE DATOS")
print("="*55)

tablas = {
    'matches'             : 'Partidos WTT crudos',
    'matches_czech'       : 'Partidos Czech crudos',
    'matches_clean'       : 'Partidos limpios',
    'matches_finalizados' : 'Partidos finalizados',
    'matches_pendientes'  : 'Partidos pendientes',
    'player_features'     : 'Features por jugador',
    'head_to_head'        : 'Head to head',
    'predicciones'        : 'Predicciones generadas',
}

try:
    for tabla, descripcion in tablas.items():
        try:
            count = pd.read_sql(
                f"SELECT COUNT(*) as n FROM {tabla}", conexion
            ).iloc[0]['n']
            print(f"  {descripcion:<25}: {int(count):>6} registros")
        except:
            print(f"  {descripcion:<25}: tabla vacia")
    conexion.close()
except:
    pass

# ── 6. REPORTE FINAL DEL PIPELINE ─────────────────────────
print(f"\n{'='*55}")
print("REPORTE DEL PIPELINE")
print("="*55)

total          = len(resultados_pipeline)
exitosos       = sum(1 for r in resultados_pipeline if r['exito'])
duracion_total = sum(r['duracion'] for r in resultados_pipeline)

for r in resultados_pipeline:
    estado = "OK   " if r['exito'] else "ERROR"
    critico_txt = "[CRITICO]" if r['critico'] else "         "
    print(f"  [{estado}] {critico_txt} {r['script']:<25} {r['duracion']}s")

print(f"  {'-'*50}")
print(f"  Scripts exitosos : {exitosos}/{total}")
print(f"  Duracion total   : {duracion_total:.2f}s")
print(f"  Log guardado en  : {log_file}")

if pipeline_detenido:
    print(f"\n  ADVERTENCIA: Pipeline detenido por error critico.")
else:
    print(f"\n  Pipeline completado exitosamente.")

print("="*55)
print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nautomation.py ejecutado correctamente")