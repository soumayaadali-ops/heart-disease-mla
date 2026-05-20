import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_loader    import charger_donnees
from preprocessing  import preparer_donnees
from evaluate       import evaluer_modele

os.makedirs("models", exist_ok=True)

# ── Chargement ────────────────────────────────────────────
df = charger_donnees("../data/raw/heart.csv")
X_train, X_test, y_train, y_test = preparer_donnees(df)

# ── Nom de l'expérience MLflow ────────────────────────────
mlflow.set_experiment("Heart_Disease_Classification")

# ── Configurations à tester ───────────────────────────────
configurations = [

    # ── Logistic Regression ──
    {"modele": LogisticRegression(C=0.1,  max_iter=1000, random_state=42),
     "nom": "LogisticRegression", "params": {"C": 0.1}},
    {"modele": LogisticRegression(C=1.0,  max_iter=1000, random_state=42),
     "nom": "LogisticRegression", "params": {"C": 1.0}},
    {"modele": LogisticRegression(C=10.0, max_iter=1000, random_state=42),
     "nom": "LogisticRegression", "params": {"C": 10.0}},

    # ── KNN ──
    {"modele": KNeighborsClassifier(n_neighbors=3),
     "nom": "KNN", "params": {"k": 3}},
    {"modele": KNeighborsClassifier(n_neighbors=5),
     "nom": "KNN", "params": {"k": 5}},
    {"modele": KNeighborsClassifier(n_neighbors=9),
     "nom": "KNN", "params": {"k": 9}},

    # ── SVM ──
    {"modele": SVC(kernel="linear", probability=True, random_state=42),
     "nom": "SVM", "params": {"kernel": "linear"}},
    {"modele": SVC(kernel="rbf",    probability=True, random_state=42),
     "nom": "SVM", "params": {"kernel": "rbf"}},
    {"modele": SVC(kernel="poly",   probability=True, random_state=42),
     "nom": "SVM", "params": {"kernel": "poly"}},

    # ── Random Forest ──
    {"modele": RandomForestClassifier(n_estimators=50,  random_state=42),
     "nom": "RandomForest", "params": {"n_estimators": 50}},
    {"modele": RandomForestClassifier(n_estimators=100, random_state=42),
     "nom": "RandomForest", "params": {"n_estimators": 100}},
    {"modele": RandomForestClassifier(n_estimators=200, random_state=42),
     "nom": "RandomForest", "params": {"n_estimators": 200}},

    # ── AdaBoost ──
    {"modele": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50, learning_rate=0.5,
        random_state=42, algorithm='SAMME'),
     "nom": "AdaBoost", "params": {"n_estimators": 50, "learning_rate": 0.5}},

    {"modele": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100, learning_rate=1.0,
        random_state=42, algorithm='SAMME'),
     "nom": "AdaBoost", "params": {"n_estimators": 100, "learning_rate": 1.0}},

    {"modele": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200, learning_rate=0.1,
        random_state=42, algorithm='SAMME'),
     "nom": "AdaBoost", "params": {"n_estimators": 200, "learning_rate": 0.1}},

    # ── XGBoost ──
    {"modele": XGBClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3,
        random_state=42, eval_metric='logloss', verbosity=0),
     "nom": "XGBoost", "params": {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}},

    {"modele": XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        random_state=42, eval_metric='logloss', verbosity=0),
     "nom": "XGBoost", "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}},

    {"modele": XGBClassifier(
        n_estimators=100, learning_rate=0.3, max_depth=5,
        random_state=42, eval_metric='logloss', verbosity=0),
     "nom": "XGBoost", "params": {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 5}},
]

# ── Entraînement + Tracking MLflow ───────────────────────
print("\n" + "="*60)
print("  DÉMARRAGE DES RUNS MLFLOW")
print("="*60)

meilleur_auc = 0
meilleur_run = None

for config in configurations:
    nom    = config["nom"]
    modele = config["modele"]
    params = config["params"]

    run_name = f"{nom}_{'_'.join(str(v) for v in params.values())}"

    with mlflow.start_run(run_name=run_name):

        # Entraînement
        modele.fit(X_train, y_train)

        # Évaluation
        metriques = evaluer_modele(modele, X_test, y_test)

        # Log paramètres
        mlflow.log_param("algorithme", nom)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log métriques
        for k, v in metriques.items():
            mlflow.log_metric(k, v)

        # Log modèle
        mlflow.sklearn.log_model(modele, artifact_path="model")

        # Affichage
        print(f"\n→ {nom} | {params}")
        for k, v in metriques.items():
            print(f"   {k:<12} : {v:.4f}")

        # Garder le meilleur
        if metriques["auc_roc"] > meilleur_auc:
            meilleur_auc = metriques["auc_roc"]
            meilleur_run = {"nom": nom, "params": params,
                            "metriques": metriques}

# ── Résumé final ──────────────────────────────────────────
print("\n" + "="*60)
print("  MEILLEUR MODÈLE")
print("="*60)
print(f"  Algorithme : {meilleur_run['nom']}")
print(f"  Paramètres : {meilleur_run['params']}")
for k, v in meilleur_run['metriques'].items():
    print(f"  {k:<12} : {v:.4f}")
print("="*60)
print("\nTous les runs sont visibles dans MLflow UI ✅")
print("Lance : mlflow ui")