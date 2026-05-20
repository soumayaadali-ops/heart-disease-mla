import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # pas de fenêtre graphique, sauvegarde directe
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("graphiques", exist_ok=True)

# ============================================================
# BLOC 1 — Chargement
# ============================================================
df = pd.read_csv("heart.csv")
print("=== Dimensions AVANT nettoyage ===")
print(f"{df.shape[0]} patients, {df.shape[1]} colonnes")

# ============================================================
# BLOC 2 — Nettoyage des doublons
# ============================================================
print(f"\nDoublons détectés : {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Doublons supprimés. Nouveau dataset : {df.shape[0]} patients")

print(f"\nValeurs manquantes :\n{df.isnull().sum()}")

# ============================================================
# BLOC 3 — Distribution de la cible
# ============================================================
counts = df['target'].value_counts()
pcts   = df['target'].value_counts(normalize=True).mul(100).round(1)

print("\n=== Distribution Target ===")
print(f"Sains   (0) : {counts[0]} patients ({pcts[0]}%)")
print(f"Malades (1) : {counts[1]} patients ({pcts[1]}%)")

plt.figure(figsize=(5, 4))
bars = plt.bar(['Sain (0)', 'Malade (1)'], counts.values,
               color=['#2196F3', '#F44336'], edgecolor='white', linewidth=1.5)
for bar, pct in zip(bars, pcts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f'{pct}%', ha='center', fontsize=12, fontweight='bold')
plt.title('Distribution des patients', fontsize=14)
plt.ylabel('Nombre de patients')
plt.tight_layout()
plt.savefig('graphiques/01_distribution_target.png', dpi=150)
plt.close()
print("Graphique sauvegardé : graphiques/01_distribution_target.png")

# ============================================================
# BLOC 4 — Distribution variables numériques
# ============================================================
cols_num = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
labels   = ['Âge', 'Pression artérielle', 'Cholestérol',
            'Fréq. cardiaque max', 'Dépression ST']

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, col, label in zip(axes, cols_num, labels):
    df[df['target']==0][col].plot(kind='hist', ax=ax, alpha=0.6,
                                   color='#2196F3', bins=20, label='Sain')
    df[df['target']==1][col].plot(kind='hist', ax=ax, alpha=0.6,
                                   color='#F44336', bins=20, label='Malade')
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8)
plt.suptitle('Distribution des variables par diagnostic', fontsize=13)
plt.tight_layout()
plt.savefig('graphiques/02_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graphique sauvegardé : graphiques/02_distributions.png")

# ============================================================
# BLOC 5 — Matrice de corrélation
# ============================================================
corr_target = df.corr()['target'].drop('target').sort_values(ascending=False)

print("\n=== Corrélation avec la maladie ===")
for col, val in corr_target.items():
    barre = "█" * int(abs(val) * 20)
    signe = "+" if val > 0 else "-"
    print(f"  {signe} {col:<12} {barre} {val:.3f}")

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, mask=mask,
            linewidths=0.5, square=True)
plt.title('Matrice de corrélation', fontsize=14)
plt.tight_layout()
plt.savefig('graphiques/03_correlation.png', dpi=150)
plt.close()
print("Graphique sauvegardé : graphiques/03_correlation.png")

# ============================================================
# RAPPORT FINAL
# ============================================================
print("\n" + "="*50)
print("  RÉSUMÉ DE L'EXPLORATION")
print("="*50)
print(f"  Patients (après nettoyage) : {df.shape[0]}")
print(f"  Colonnes                   : {df.shape[1]}")
print(f"  Valeurs manquantes         : {df.isnull().sum().sum()}")
print(f"  Sains   (0)                : {counts[0]} ({pcts[0]}%)")
print(f"  Malades (1)                : {counts[1]} ({pcts[1]}%)")
print()
print("  Top 5 variables liées à la maladie :")
for col, val in corr_target.abs().sort_values(ascending=False).head(5).items():
    print(f"    → {col:<12} : {val:.3f}")
print("="*50)
print("\nExploration terminée ! Graphiques dans /graphiques")