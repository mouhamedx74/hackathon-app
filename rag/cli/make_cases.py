"""Génère data/cases.csv : 30 cas FICTIFS de démarrage (15 durables, 15 buzz, dont cas limites).
À REMPLACER / faire valider par Product-Insights avec de vraies tendances passées."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
cols = ["persistence", "growth", "diversity", "diffusion", "adoption"]

def clip(x): return np.clip(np.round(x), 0, 100)

rows = []
for i in range(15):   # durables : piliers élevés, peu de pic viral
    p = clip(rng.normal([82, 75, 78, 68, 60], 9))
    rows.append([f"durable_{i+1:02d}", *p, round(float(rng.choice([0, 0, 0, .2])), 2),
                 float(clip(rng.normal(78, 10))), float(clip(rng.normal(70, 12))), "durable"])
for i in range(15):   # buzz : croissance forte mais peu persistant/diversifié, pic viral
    p = clip(rng.normal([30, 70, 30, 25, 15], 10))
    rows.append([f"buzz_{i+1:02d}", *p, round(float(rng.uniform(.5, 1)), 2),
                 float(clip(rng.normal(60, 15))), float(clip(rng.normal(50, 15))), "buzz"])
# 4 cas limites volontairement ambigus
rows[3][1:6] = [58, 62, 55, 45, 30]; rows[3][6] = 0.3
rows[9][1:6] = [70, 55, 52, 40, 25]; rows[9][6] = 0.0
rows[18][1:6] = [55, 80, 50, 40, 22]; rows[18][6] = 0.4
rows[25][1:6] = [62, 75, 60, 50, 35]; rows[25][6] = 0.7

df = pd.DataFrame(rows, columns=["trend", *cols, "viral_spike", "fit", "timing", "label"])
df.to_csv("data/cases.csv", index=False)
print(df.label.value_counts().to_dict(), "->", "data/cases.csv")
