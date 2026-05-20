import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                              accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs("../graphiques", exist_ok=True)

# ── CHARGEMENT ────────────────────────────────────────────
df = pd.read_csv("../data/raw/heart.csv").drop_duplicates()
X  = df.drop(columns=['target'])
y  = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler    = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

mlflow.set_experiment("Heart_Disease_Classification")

# ── 3 RUNS AVEC ARTEFACTS ─────────────────────────────────
configs = [
    {"run_name": "rf_50trees_depth3",
     "model": RandomForestClassifier(n_estimators=50,  max_depth=3,    random_state=42),
     "params": {"model_type":"RandomForest","n_estimators":50,  "max_depth":3,    "test_size":0.2}},
    {"run_name": "rf_200trees_depth10",
     "model": RandomForestClassifier(n_estimators=200, max_depth=10,   random_state=42),
     "params": {"model_type":"RandomForest","n_estimators":200, "max_depth":10,   "test_size":0.2}},
    {"run_name": "rf_100trees_depthNone",
     "model": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
     "params": {"model_type":"RandomForest","n_estimators":100, "max_depth":"None","test_size":0.2}},
]

print("\n" + "="*60)
print("  PARTIE 1 — LOGGING ARTEFACTS MLFLOW")
print("="*60)

for cfg in configs:
    with mlflow.start_run(run_name=cfg["run_name"]):

        # Entraînement
        model = cfg["model"]
        model.fit(X_train_s, y_train)
        y_pred  = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        # Log params
        mlflow.log_params(cfg["params"])

        # Log métriques
        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "f1_score":  round(f1_score(y_test, y_pred),         4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba),   4),
        }
        mlflow.log_metrics(metrics)

        # Log modèle
        mlflow.sklearn.log_model(model, artifact_path="model")

        # ── ARTEFACT 1 : Matrice de confusion ──
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax,
            display_labels=["Sain","Malade"],
            colorbar=False, cmap="Blues"
        )
        ax.set_title(f"Matrice de confusion — {cfg['run_name']}")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        # ── ARTEFACT 2 : Rapport de classification ──
        report = classification_report(
            y_test, y_pred,
            target_names=["Sain","Malade"]
        )
        with open("classification_report.txt", "w") as f:
            f.write(f"Run : {cfg['run_name']}\n\n")
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        print(f"\n→ {cfg['run_name']}")
        for k, v in metrics.items():
            print(f"   {k:<12} : {v:.4f}")
        print("   Artefacts loggés : confusion_matrix.png + classification_report.txt ✅")

print("\nPartie 1 terminée ✅")