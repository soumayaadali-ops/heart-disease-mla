# 🫀 Heart Disease Classification — Projet MLA

Classification des patients à risque de maladies cardiaques
à partir de leurs paramètres biologiques et médicaux.

## Dataset
Heart Disease Dataset — Kaggle (johnsmith88) — 302 patients, 14 variables

## Stack technique
- **Backend** : Python, FastAPI, Scikit-learn, MLflow
- **Frontend** : React + Tailwind CSS + Recharts

## Algorithmes implémentés (Tâche 3)
| Modèle | Meilleur paramètre | Accuracy | Recall |
|---|---|---|---|
| SVM | kernel=poly | 81.97% | 90.91% |
| SVM | kernel=linear | 81.97% | 87.88% |
| Logistic Regression | C=1.0 | 80.33% | 84.85% |
| KNN | k=9 | 80.33% | 87.88% |
| Random Forest | n=100 | 75.41% | 78.79% |

## Lancer le projet

### Backend
```bash
cd backend
pip install -r requirements.txt
python train.py      # entraînement + MLflow
uvicorn app:app --reload  # API REST
```

### MLflow UI
```bash
mlflow ui
# Ouvrir http://127.0.0.1:5000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Ouvrir http://localhost:5173
```

## Structure du projet
```
heart-disease-mla/
├── backend/          # API FastAPI + MLflow
├── frontend/         # Interface React
├── data/raw/         # Dataset brut
└── mlruns/           # Runs MLflow
```
