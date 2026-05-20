import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs("../graphiques", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)

# ── CHARGEMENT ────────────────────────────────────────────
df = pd.read_csv("../data/raw/heart.csv").drop_duplicates()
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

mlflow.set_experiment("Heart_Disease_Tache4")

# ════════════════════════════════════════════════════════
# QUESTION 1 — Feature Importance
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Q1 — IMPORTANCE DES FEATURES")
print("="*60)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_s, y_train)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nImportance de chaque variable :")
for col, val in importances.items():
    barre = "█" * int(val * 50)
    print(f"  {col:<12} {barre} {val:.4f}")

top3 = importances.head(3)
print(f"\nTop 3 variables : {top3.index.tolist()}")

# Graphique
plt.figure(figsize=(10, 6))
colors = ['#E74C3C' if i < 3 else '#3498DB' for i in range(len(importances))]
plt.barh(importances.index[::-1], importances.values[::-1], color=colors[::-1])
plt.xlabel("Importance")
plt.title("Importance des features — Random Forest")
plt.axvline(x=importances.mean(), color='orange', linestyle='--', label=f'Moyenne ({importances.mean():.3f})')
plt.legend()
plt.tight_layout()
plt.savefig('../graphiques/07_feature_importance.png', dpi=150)
plt.close()
print("Graphique sauvegardé : graphiques/07_feature_importance.png")

# ════════════════════════════════════════════════════════
# QUESTION 2 — Stabilité avec différents random_state
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Q2 — STABILITÉ DES PRÉDICTIONS")
print("="*60)

random_states = [0, 1, 7, 21, 42, 99, 123, 200, 500, 999]
accuracies = []

for rs in random_states:
    rf_rs = RandomForestClassifier(n_estimators=100, random_state=rs)
    rf_rs.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, rf_rs.predict(X_test_s))
    accuracies.append(acc)
    print(f"  random_state={rs:<5} → Accuracy = {acc*100:.2f}%")

print(f"\n  Moyenne   : {np.mean(accuracies)*100:.2f}%")
print(f"  Écart-type: {np.std(accuracies)*100:.2f}%")
print(f"  Min       : {np.min(accuracies)*100:.2f}%")
print(f"  Max       : {np.max(accuracies)*100:.2f}%")

variabilite = np.max(accuracies) - np.min(accuracies)
if variabilite < 0.05:
    print(f"\n  → Modèle STABLE (variabilité = {variabilite*100:.2f}%)")
else:
    print(f"\n  → Modèle INSTABLE (variabilité = {variabilite*100:.2f}%)")

plt.figure(figsize=(8, 4))
plt.plot(random_states, [a*100 for a in accuracies], 'o-', color='#3498DB', linewidth=2)
plt.axhline(y=np.mean(accuracies)*100, color='orange', linestyle='--',
            label=f'Moyenne = {np.mean(accuracies)*100:.2f}%')
plt.xlabel("random_state")
plt.ylabel("Accuracy (%)")
plt.title("Stabilité du Random Forest selon random_state")
plt.legend()
plt.tight_layout()
plt.savefig('../graphiques/08_stabilite.png', dpi=150)
plt.close()
print("Graphique sauvegardé : graphiques/08_stabilite.png")

# ════════════════════════════════════════════════════════
# QUESTION 3 — Analyse des erreurs
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Q3 — ANALYSE DES ERREURS")
print("="*60)

y_pred = rf.predict(X_test_s)
erreurs_idx = np.where(y_pred != y_test.values)[0]

print(f"\nNombre d'erreurs : {len(erreurs_idx)} / {len(y_test)}")
print(f"Taux d'erreur    : {len(erreurs_idx)/len(y_test)*100:.1f}%\n")

X_test_reset = X_test_s.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

faux_positifs = 0
faux_negatifs = 0

for i, idx in enumerate(erreurs_idx[:5]):
    reel   = y_test_reset[idx]
    predit = y_pred[idx]
    type_erreur = "Faux Positif (Sain → Malade)" if predit == 1 else "Faux Négatif (Malade → Sain)"
    if predit == 1: faux_positifs += 1
    else: faux_negatifs += 1

    print(f"  Erreur {i+1} — {type_erreur}")
    print(f"    age={X_test_reset.loc[idx,'age']:.0f}  "
          f"thalach={X_test_reset.loc[idx,'thalach']:.2f}  "
          f"ca={X_test_reset.loc[idx,'ca']:.2f}  "
          f"oldpeak={X_test_reset.loc[idx,'oldpeak']:.2f}  "
          f"cp={X_test_reset.loc[idx,'cp']:.2f}")

print(f"\n  Faux Positifs (Sain classé Malade) : {faux_positifs}")
print(f"  Faux Négatifs (Malade classé Sain) : {faux_negatifs}")

# ════════════════════════════════════════════════════════
# QUESTION 4 — Biais et Variance
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Q4 — BIAIS ET VARIANCE")
print("="*60)

configurations = [
    {"n_estimators": 10,  "max_depth": 2},
    {"n_estimators": 10,  "max_depth": 5},
    {"n_estimators": 50,  "max_depth": 3},
    {"n_estimators": 50,  "max_depth": None},
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 200, "max_depth": None},
]

resultats_bv = []
print(f"\n{'n_est':>6} {'max_d':>6} {'Train%':>8} {'Test%':>8} {'Biais':>10} {'Variance':>10} {'Diagnostic':>15}")
print("-"*70)

for config in configurations:
    with mlflow.start_run(run_name=f"RF_n{config['n_estimators']}_d{config['max_depth']}"):
        rf_bv = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            random_state=42
        )
        rf_bv.fit(X_train_s, y_train)

        train_acc = accuracy_score(y_train, rf_bv.predict(X_train_s))
        test_acc  = accuracy_score(y_test,  rf_bv.predict(X_test_s))
        biais     = 1 - train_acc
        variance  = train_acc - test_acc

        if variance > 0.1:
            diagnostic = "Overfitting"
        elif biais > 0.25:
            diagnostic = "Underfitting"
        else:
            diagnostic = "Équilibré"

        mlflow.log_params(config)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy",  test_acc)
        mlflow.log_metric("biais",          biais)
        mlflow.log_metric("variance",       variance)

        print(f"{config['n_estimators']:>6} "
              f"{str(config['max_depth']):>6} "
              f"{train_acc*100:>7.2f}% "
              f"{test_acc*100:>7.2f}% "
              f"{biais:>10.4f} "
              f"{variance:>10.4f} "
              f"{diagnostic:>15}")

        resultats_bv.append({
            "n_estimators": config['n_estimators'],
            "max_depth":    str(config['max_depth']),
            "train_acc":    round(train_acc*100, 2),
            "test_acc":     round(test_acc*100, 2),
            "biais":        round(biais, 4),
            "variance":     round(variance, 4),
            "diagnostic":   diagnostic
        })

# Graphique biais-variance
df_bv = pd.DataFrame(resultats_bv)
labels = [f"n={r['n_estimators']}\nd={r['max_depth']}" for _, r in df_bv.iterrows()]

plt.figure(figsize=(12, 5))
x = np.arange(len(df_bv))
plt.bar(x - 0.2, df_bv['train_acc'], 0.35, label='Train Accuracy', color='#3498DB')
plt.bar(x + 0.2, df_bv['test_acc'],  0.35, label='Test Accuracy',  color='#E74C3C')
plt.xticks(x, labels, fontsize=8)
plt.ylabel("Accuracy (%)")
plt.title("Biais-Variance selon les hyperparamètres — Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig('../graphiques/09_biais_variance.png', dpi=150)
plt.close()
print("\nGraphique sauvegardé : graphiques/09_biais_variance.png")

# ════════════════════════════════════════════════════════
# QUESTION 5 — Comparaison Random Forest vs Arbre de décision
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Q5 — RANDOM FOREST vs ARBRE DE DÉCISION")
print("="*60)

profondeurs = [3, 5, 10, None]
print(f"\n{'Modèle':<30} {'max_depth':>10} {'Train%':>8} {'Test%':>8} {'F1':>8}")
print("-"*65)

for depth in profondeurs:
    with mlflow.start_run(run_name=f"DT_depth_{depth}"):
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train_s, y_train)
        t_acc = accuracy_score(y_train, dt.predict(X_train_s))
        v_acc = accuracy_score(y_test,  dt.predict(X_test_s))
        f1    = f1_score(y_test, dt.predict(X_test_s))
        mlflow.log_params({"modele": "DecisionTree", "max_depth": str(depth)})
        mlflow.log_metrics({"train_accuracy": t_acc, "test_accuracy": v_acc, "f1": f1})
        print(f"  {'Decision Tree':<28} {str(depth):>10} {t_acc*100:>7.2f}% {v_acc*100:>7.2f}% {f1:>7.4f}")

for depth in profondeurs:
    with mlflow.start_run(run_name=f"RF_vs_DT_depth_{depth}"):
        rf_cmp = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        rf_cmp.fit(X_train_s, y_train)
        t_acc = accuracy_score(y_train, rf_cmp.predict(X_train_s))
        v_acc = accuracy_score(y_test,  rf_cmp.predict(X_test_s))
        f1    = f1_score(y_test, rf_cmp.predict(X_test_s))
        mlflow.log_params({"modele": "RandomForest", "max_depth": str(depth)})
        mlflow.log_metrics({"train_accuracy": t_acc, "test_accuracy": v_acc, "f1": f1})
        print(f"  {'Random Forest':<28} {str(depth):>10} {t_acc*100:>7.2f}% {v_acc*100:>7.2f}% {f1:>7.4f}")

print("\n" + "="*60)
print("  TÂCHE 4 TERMINÉE ✅")
print("  Graphiques dans /graphiques/")
print("  Runs visibles dans MLflow UI → mlflow ui")
print("="*60)