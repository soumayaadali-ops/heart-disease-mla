import numpy as np
import pandas as pd
import mlflow
from scipy import stats
from sklearn.model_selection import train_test_split
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ── CHARGEMENT ────────────────────────────────────────────
df = pd.read_csv("../data/raw/heart.csv").drop_duplicates()
X  = df.drop(columns=['target'])
y  = df['target']

X_train, X_test, _, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── SIMULATION DU DRIFT ───────────────────────────────────
X_prod = X_test.copy()
num_cols = X_prod.select_dtypes(include=np.number).columns

for col in num_cols[:3]:
    X_prod[col] = X_prod[col] * 1.6 + np.random.normal(0, 0.5, len(X_prod))

print("\n" + "="*60)
print("  PARTIE 6 — DÉTECTION DATA DRIFT")
print("="*60)
print("\nSimulation drift : multiplication x1.6 + bruit gaussien")
print(f"{'Feature':<15} {'Moy. Train':>12} {'Moy. Prod':>12} {'Drifté':>10}")
print("-"*50)
for col in num_cols[:3]:
    print(f"  {col:<13} {X_train[col].mean():>12.3f} "
          f"{X_prod[col].mean():>12.3f}  {'OUI ⚠️':>10}")

# ── EVIDENTLY ─────────────────────────────────────────────
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric

    mlflow.set_experiment("monitoring_drift")

    with mlflow.start_run(run_name="drift_check_v1"):

        # Rapport HTML complet
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=X_train, current_data=X_prod)
        report.save_html("drift_report.html")
        mlflow.log_artifact("drift_report.html")
        print("\n✅ Rapport Evidently HTML sauvegardé : drift_report.html")

        # Score numérique
        score_report = Report(metrics=[DatasetDriftMetric()])
        score_report.run(reference_data=X_train, current_data=X_prod)
        result = score_report.as_dict()

        drift_share   = result['metrics'][0]['result']['drift_share']
        n_drifted     = result['metrics'][0]['result']['number_of_drifted_columns']
        n_total       = result['metrics'][0]['result']['number_of_columns']
        dataset_drift = result['metrics'][0]['result']['dataset_drift']

        mlflow.log_metric("drift_share",      drift_share)
        mlflow.log_metric("drifted_columns",  n_drifted)
        mlflow.log_metric("total_columns",    n_total)
        mlflow.log_metric("dataset_drifted",  int(dataset_drift))

        print(f"\n📊 Evidently Results :")
        print(f"   Drift share      : {drift_share:.2%}")
        print(f"   Colonnes driftées: {n_drifted}/{n_total}")
        print(f"   Dataset drifté   : {'OUI' if dataset_drift else 'NON'}")

        # ── KS-TEST PAR FEATURE ───────────────────────────
        print(f"\n📊 KS-Test par feature :")
        print(f"{'Feature':<15} {'KS stat':>10} {'p-value':>10} {'Drifté':>10}")
        print("-"*48)

        ks_results = []
        for col in X_train.select_dtypes(include='number').columns:
            stat, pvalue = stats.ks_2samp(X_train[col], X_prod[col])
            drifted = pvalue < 0.05
            ks_results.append({
                'feature': col, 'ks_stat': round(stat, 4),
                'p_value': round(pvalue, 4), 'drifted': drifted
            })
            mlflow.log_metric(f"ks_pvalue_{col}", pvalue)
            flag = "⚠️ OUI" if drifted else "✅ NON"
            print(f"  {col:<13} {stat:>10.4f} {pvalue:>10.4f} {flag:>10}")

        df_ks = pd.DataFrame(ks_results)
        df_ks.to_csv("ks_drift_results.csv", index=False)
        mlflow.log_artifact("ks_drift_results.csv")
        print("\n✅ Résultats KS-test sauvegardés : ks_drift_results.csv")

        # ── DÉCLENCHEMENT AUTO ────────────────────────────
        SEUIL_DRIFT = 0.30
        SEUIL_WARN  = 0.15

        print(f"\n🔁 Logique de ré-entraînement :")
        print(f"   Drift share   : {drift_share:.2%}")
        print(f"   Seuil alerte  : {SEUIL_WARN:.0%}")
        print(f"   Seuil critique: {SEUIL_DRIFT:.0%}")

        if drift_share > SEUIL_DRIFT:
            print(f"\n🚨 CRITIQUE : drift {drift_share:.2%} > {SEUIL_DRIFT:.0%}")
            print("   → Ré-entraînement déclenché !")
            mlflow.log_metric("retrain_triggered", 1)
        elif drift_share > SEUIL_WARN:
            print(f"\n⚠️  ALERTE : drift {drift_share:.2%} — surveillance renforcée")
            mlflow.log_metric("retrain_triggered", 0)
        else:
            print(f"\n✅ OK : drift {drift_share:.2%} — modèle stable")
            mlflow.log_metric("retrain_triggered", 0)

except ImportError:
    print("\n⚠️ Evidently non installé — lance : pip install evidently")
    print("KS-test uniquement :\n")

    mlflow.set_experiment("monitoring_drift")
    with mlflow.start_run(run_name="drift_kstest_only"):
        ks_results = []
        for col in X_train.select_dtypes(include='number').columns:
            stat, pvalue = stats.ks_2samp(X_train[col], X_prod[col])
            drifted = pvalue < 0.05
            ks_results.append({'feature':col,'ks_stat':round(stat,4),
                                'p_value':round(pvalue,4),'drifted':drifted})
            mlflow.log_metric(f"ks_pvalue_{col}", pvalue)

        df_ks = pd.DataFrame(ks_results)
        df_ks.to_csv("ks_drift_results.csv", index=False)
        mlflow.log_artifact("ks_drift_results.csv")
        print(df_ks.to_string(index=False))

print("\nPartie 6 terminée ✅")