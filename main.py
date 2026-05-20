from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os

app = FastAPI(title="Heart Disease API", version="1.0")

# Autoriser React à communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CHARGEMENT DES MODÈLES AU DÉMARRAGE
# ============================================================
modeles = {}
noms_modeles = [
    "Logistic_Regression",
    "SVM",
    "Random_Forest",
    "KNN",
    "Neural_Network",
    "AdaBoost",
    "XGBoost"
]

for nom in noms_modeles:
    path = f"data/modeles/{nom}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            modeles[nom] = pickle.load(f)
        print(f"✅ Modèle chargé : {nom}")

with open("data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler chargé")

resultats_df = pd.read_csv("data/resultats.csv")

# ============================================================
# SCHÉMA D'UN PATIENT
# ============================================================
class Patient(BaseModel):
    age:      int
    sex:      int
    cp:       int
    trestbps: int
    chol:     int
    fbs:      int
    restecg:  int
    thalach:  int
    exang:    int
    oldpeak:  float
    slope:    int
    ca:       int
    thal:     int

# ============================================================
# ROUTES API
# ============================================================

# Route test
@app.get("/")
def accueil():
    return {"message": "Heart Disease API opérationnelle ✅"}

# Liste des modèles + leurs scores
@app.get("/models")
def get_models():
    return resultats_df.to_dict(orient="records")

# Prédiction pour un patient
@app.post("/predict/{nom_modele}")
def predict(nom_modele: str, patient: Patient):

    if nom_modele not in modeles:
        return {"erreur": f"Modèle '{nom_modele}' introuvable"}

    # Préparer les données du patient
    data = pd.DataFrame([patient.dict()])
    data_scaled = scaler.transform(data)

    # Prédiction
    modele  = modeles[nom_modele]
    pred    = modele.predict(data_scaled)[0]
    proba   = modele.predict_proba(data_scaled)[0]

    return {
        "modele":     nom_modele,
        "prediction": int(pred),
        "diagnostic": "Malade" if pred == 1 else "Sain",
        "probabilite_sain":   round(float(proba[0]) * 100, 2),
        "probabilite_malade": round(float(proba[1]) * 100, 2),
    }

# Résultats complets de tous les modèles
@app.get("/results")
def get_results():
    return resultats_df.to_dict(orient="records")