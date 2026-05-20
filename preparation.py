import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

os.makedirs("data", exist_ok=True)

# ============================================================
# ÉTAPE 1 — Charger et nettoyer
# ============================================================
df = pd.read_csv("heart.csv")
df = df.drop_duplicates()
print(f"Dataset chargé : {df.shape[0]} patients, {df.shape[1]} colonnes")

# ============================================================
# ÉTAPE 2 — Séparer features et cible
# ============================================================
X = df.drop(columns=['target'])   # tout sauf la cible
y = df['target']                   # 0 = sain, 1 = malade

print(f"\nFeatures (X) : {X.shape}")
print(f"Cible    (y) : {y.shape}")

# ============================================================
# ÉTAPE 3 — Split Train 80% / Test 20%
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain : {X_train.shape[0]} patients")
print(f"Test  : {X_test.shape[0]} patients")

# ============================================================
# ÉTAPE 4 — Normalisation
# ============================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

print("\nNormalisation effectuée ✅")

# ============================================================
# ÉTAPE 5 — Sauvegarde
# ============================================================
X_train_scaled.to_csv("data/X_train.csv", index=False)
X_test_scaled.to_csv("data/X_test.csv",   index=False)
y_train.to_csv("data/y_train.csv",         index=False)
y_test.to_csv("data/y_test.csv",           index=False)

with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nFichiers sauvegardés dans /data :")
print("  ✅ X_train.csv")
print("  ✅ X_test.csv")
print("  ✅ y_train.csv")
print("  ✅ y_test.csv")
print("  ✅ scaler.pkl")
print("\nPréparation terminée ! Prêt pour l'entraînement des modèles 🚀")