import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

os.makedirs("../graphiques", exist_ok=True)

# Charger les données
X_train = pd.read_csv("../data/processed/X_train.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()

couleurs = {0: '#2196F3', 1: '#F44336'}
labels   = {0: 'Sain', 1: 'Malade'}

# ── PCA ──────────────────────────────────────────────────
pca    = PCA(n_components=2)
X_pca  = pca.fit_transform(X_train)
var    = pca.explained_variance_ratio_

plt.figure(figsize=(7, 5))
for classe in [0, 1]:
    mask = y_train == classe
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=couleurs[classe], label=labels[classe],
                alpha=0.6, edgecolors='white', linewidth=0.5, s=60)
plt.title(f'PCA — Variance expliquée : {var[0]*100:.1f}% + {var[1]*100:.1f}%')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.legend()
plt.tight_layout()
plt.savefig('../graphiques/05_pca.png', dpi=150)
plt.close()
print("PCA sauvegardée : graphiques/05_pca.png")

# ── t-SNE ────────────────────────────────────────────────
tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_train)

plt.figure(figsize=(7, 5))
for classe in [0, 1]:
    mask = y_train == classe
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                c=couleurs[classe], label=labels[classe],
                alpha=0.6, edgecolors='white', linewidth=0.5, s=60)
plt.title('t-SNE — Visualisation 2D des patients')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.tight_layout()
plt.savefig('../graphiques/06_tsne.png', dpi=150)
plt.close()
print("t-SNE sauvegardée : graphiques/06_tsne.png")

print("\nLes deux graphiques sont dans le dossier graphiques/ ✅")