# ============================================
# Entraîner Random Forest pour prédire Smax
# à partir de resume_Smax_final.csv + Y_Smax.csv
# + sauvegarde best_model.joblib pour l'appli Dash
# ============================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ----------------------------
# 1) Charger X et y
# ----------------------------
X_df = pd.read_csv("resume_Smax_final.csv")
y_df = pd.read_csv("Y_Smax.csv")

# Nettoyage noms colonnes
X_df.columns = [c.strip() for c in X_df.columns]
y_df.columns = [c.strip() for c in y_df.columns]

# Vérifs
if "Nom_Fichier" not in X_df.columns:
    raise ValueError("resume_Smax_final.csv doit contenir la colonne 'Nom_Fichier'")
if "Nom_Fichier" not in y_df.columns or "Y_Smax" not in y_df.columns:
    raise ValueError("Y_Smax.csv doit contenir les colonnes 'Nom_Fichier' et 'Y_Smax'")

# ----------------------------
# 2) Fusion sur Nom_Fichier
# ----------------------------
df = pd.merge(X_df, y_df, on="Nom_Fichier", how="inner")

if df.empty:
    raise ValueError("Fusion vide : les 'Nom_Fichier' ne matchent pas entre X et Y.")

print("✅ Fusion OK :", df.shape)
print("Colonnes:", df.columns.tolist())

# ----------------------------
# 3) Séparer X / y
# ----------------------------
y = df["Y_Smax"].astype(float)

X = df.drop(columns=["Y_Smax"])

# On retire l'identifiant
X = X.drop(columns=["Nom_Fichier"], errors="ignore")

# Garder uniquement numérique
X = X.select_dtypes(include=[np.number])

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# ----------------------------
# 4) Split pour évaluation
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ----------------------------
# 5) Entraîner RF (modèle de travail)
# ----------------------------
rf = RandomForestRegressor(
    n_estimators=600,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ----------------------------
# 6) Évaluer
# ----------------------------
pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\n=== RANDOM FOREST (split 70/30) ===")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.4f}")

# ----------------------------
# 7) Modèle FINAL (fit sur tout)
# ----------------------------
rf_final = RandomForestRegressor(
    n_estimators=600,
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X, y)

# ----------------------------
# 8) Sauvegarde best_model.joblib
# ----------------------------
joblib.dump(
    {
        "model": rf_final,
        "feature_names": X.columns.tolist(),
        "target": "Y_Smax"
    },
    "best_model.joblib"
)

print("\n✅ best_model.joblib créé.")
print("Nb features :", len(X.columns))
print("Exemples features :", X.columns.tolist()[:10])
