"""Évalue V1 (scoring à règles) et V2 (régression logistique) sur data/cases.csv."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from app.scoring import durability_score, PILLARS, DURABLE_THRESHOLD

path = sys.argv[1] if len(sys.argv) > 1 else "data/cases.csv"
df = pd.read_csv(path)
y = (df["label"] == "durable").astype(int)

def report(name, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"\n== {name} ==")
    print(f"matrice de confusion  [[TN={tn} FP={fp}] [FN={fn} TP={tp}]]")
    print(f"accuracy={accuracy_score(y, pred):.2f}  precision={precision_score(y, pred):.2f}  "
          f"recall={recall_score(y, pred):.2f}  F1={f1_score(y, pred):.2f}")
    return pred

# V1 : scoring à règles
df["durability"] = [durability_score(r[PILLARS].to_dict(), r["viral_spike"]) for _, r in df.iterrows()]
report(f"V1 scoring (seuil {DURABLE_THRESHOLD})", (df["durability"] >= DURABLE_THRESHOLD).astype(int))

# V2 : régression logistique, validation croisée (évite le surapprentissage sur 30 cas)
X = df[PILLARS + ["viral_spike"]]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(5, shuffle=True, random_state=42)
report("V2 régression logistique (CV 5 plis)", cross_val_predict(model, X, y, cv=cv))

fails = df[(df["durability"] >= DURABLE_THRESHOLD).astype(int) != y]
print("\nCas mal classés par V1 :")
print(fails[["trend", "durability", "label"]].to_string(index=False) if len(fails) else "aucun")
