import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

os.makedirs("data/modeles", exist_ok=True)
os.makedirs("graphiques",   exist_ok=True)

# ── CHARGEMENT ────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print(f"Train : {X_train.shape[0]} patients")
print(f"Test  : {X_test.shape[0]} patients")

# ── DÉFINITION DES 7 MODÈLES ──────────────────────────────
modeles = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Neural Network":      MLPClassifier(hidden_layer_sizes=(64, 32),
                                         max_iter=1000, random_state=42),
    "AdaBoost":            AdaBoostClassifier(
                               estimator=DecisionTreeClassifier(max_depth=1),
                               n_estimators=50, learning_rate=0.5,
                               random_state=42),
    "XGBoost":             XGBClassifier(
                               n_estimators=50, learning_rate=0.1,
                               max_depth=3, random_state=42,
                               eval_metric='logloss', verbosity=0),
}

# ── ENTRAÎNEMENT ET ÉVALUATION ────────────────────────────
resultats = []

print("\n" + "="*55)
print("  ENTRAÎNEMENT DES 7 MODÈLES")
print("="*55)

for nom, modele in modeles.items():
    print(f"\n→ {nom} en cours...")

    modele.fit(X_train, y_train)

    y_pred  = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    resultats.append({
        "Modèle":   nom,
        "Accuracy": round(acc * 100, 2),
        "F1-Score": round(f1  * 100, 2),
        "AUC-ROC":  round(auc * 100, 2),
    })

    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   F1-Score : {f1*100:.2f}%")
    print(f"   AUC-ROC  : {auc*100:.2f}%")

    with open(f"data/modeles/{nom.replace(' ', '_')}.pkl", "wb") as f:
        pickle.dump(modele, f)

# ── TABLEAU COMPARATIF ────────────────────────────────────
df_resultats = pd.DataFrame(resultats).sort_values("Accuracy", ascending=False)

print("\n" + "="*55)
print("  COMPARAISON DES 7 MODÈLES")
print("="*55)
print(df_resultats.to_string(index=False))
print("="*55)
print(f"\n🏆 Meilleur modèle : {df_resultats.iloc[0]['Modèle']}")

df_resultats.to_csv("data/resultats.csv", index=False)

# ── GRAPHIQUE COMPARATIF ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x     = np.arange(len(df_resultats))
width = 0.25
noms  = df_resultats['Modèle'].tolist()

bars1 = ax.bar(x - width, df_resultats['Accuracy'], width,
               label='Accuracy', color='#2196F3')
bars2 = ax.bar(x,          df_resultats['F1-Score'], width,
               label='F1-Score', color='#4CAF50')
bars3 = ax.bar(x + width,  df_resultats['AUC-ROC'],  width,
               label='AUC-ROC', color='#FF9800')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{bar.get_height():.1f}',
                ha='center', va='bottom', fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(noms, rotation=15, ha='right')
ax.set_ylim(60, 105)
ax.set_ylabel('Score (%)')
ax.set_title('Comparaison des 7 modèles ML', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('graphiques/04_comparaison_modeles.png', dpi=150)
plt.close()
print("\nGraphique sauvegardé ✅")