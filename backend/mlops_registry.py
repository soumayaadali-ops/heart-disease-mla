import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# ════════════════════════════════════════════════════════
# PARTIE 2 — REQUÊTES PROGRAMMATIQUES
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PARTIE 2 — CLIENT MLFLOW PROGRAMMATIQUE")
print("="*60)

experiment = client.get_experiment_by_name("Heart_Disease_Classification")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=5
)

print("\nTop 5 runs par Accuracy :")
print(f"{'Run Name':<30} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
print("-"*62)
for run in runs:
    name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
    acc  = run.data.metrics.get("accuracy", 0)
    f1   = run.data.metrics.get("f1_score", 0)
    auc  = run.data.metrics.get("roc_auc",  run.data.metrics.get("auc_roc", 0))
    print(f"  {name:<28} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")

best_run = runs[0]
best_run_id = best_run.info.run_id
best_name   = best_run.data.tags.get("mlflow.runName", best_run_id[:8])
best_acc    = best_run.data.metrics.get("accuracy", 0)

print(f"\n🏆 Meilleur run : {best_name}")
print(f"   Run ID   : {best_run_id}")
print(f"   Accuracy : {best_acc:.4f}")
print(f"   Params   : {best_run.data.params}")

# ════════════════════════════════════════════════════════
# PARTIE 3 — MODEL REGISTRY
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PARTIE 3 — MODEL REGISTRY")
print("="*60)

MODEL_NAME = "HeartDisease_Production"

# Enregistrement du meilleur modèle
model_uri = f"runs:/{best_run_id}/model"
print(f"\nEnregistrement du modèle depuis : {model_uri}")

registered = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)
print(f"✅ Modèle enregistré — Version : {registered.version}")

# Description + tag
client.update_registered_model(
    name=MODEL_NAME,
    description="Modèle de classification maladies cardiaques — version optimisée"
)
client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered.version,
    key="validated_by",
    value="equipe_data"
)
print("✅ Description et tag ajoutés")

# Promotion en Staging
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=registered.version,
    stage="Staging",
    archive_existing_versions=False
)
print(f"✅ Modèle v{registered.version} promu en Staging")

# Validation avant Production
SEUIL_PRODUCTION = 0.75
print(f"\nValidation — seuil Production : {SEUIL_PRODUCTION*100:.0f}%")
print(f"Accuracy du meilleur run      : {best_acc*100:.2f}%")

if best_acc >= SEUIL_PRODUCTION:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Modèle v{registered.version} promu en PRODUCTION !")
else:
    print(f"⚠️ Modèle non promu : accuracy {best_acc:.3f} < seuil {SEUIL_PRODUCTION}")

print("\nPartie 3 terminée ✅")
print("Lance mlflow ui pour voir le Registry → http://127.0.0.1:5000")