import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd

conn = sqlite3.connect('base_ping_pong.sqlite')
df = pd.read_sql('SELECT puntos_local_s1, puntos_visitante_s1, puntos_local_s2, puntos_visitante_s2, puntos_local_s3, puntos_visitante_s3, puntos_local_s4, puntos_visitante_s4, puntos_local_s5, puntos_visitante_s5 FROM matches_finalizados WHERE puntos_local_s1 > 0', conn)
conn.close()

for i in range(1, 6):
    col_l = f'puntos_local_s{i}'
    col_v = f'puntos_visitante_s{i}'
    if col_l in df.columns:
        total = pd.to_numeric(df[col_l], errors='coerce') + pd.to_numeric(df[col_v], errors='coerce')
        total = total.dropna()
        if len(total) > 100:
            bajo  = (total < 17.5).sum()
            medio = ((total >= 17.5) & (total <= 19.5)).sum()
            alto  = (total > 19.5).sum()
            print(f"Set {i}: promedio={total.mean():.1f} | bajo 17.5={bajo/len(total)*100:.1f}% | entre 17.5-19.5={medio/len(total)*100:.1f}% | alto 19.5={alto/len(total)*100:.1f}%")