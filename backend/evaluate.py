from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)

def evaluer_modele(modele, X_test, y_test):
    y_pred  = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),          4),
        "precision": round(precision_score(y_test, y_pred),          4),
        "recall":    round(recall_score(y_test, y_pred),             4),
        "f1_score":  round(f1_score(y_test, y_pred),                 4),
        "auc_roc":   round(roc_auc_score(y_test, y_proba),           4),
    }