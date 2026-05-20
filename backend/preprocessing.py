import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def preparer_donnees(df):
    os.makedirs("data/processed", exist_ok=True)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    # Sauvegarder
    X_train_s.to_csv("../data/processed/X_train.csv", index=False)
    X_test_s.to_csv("../data/processed/X_test.csv",   index=False)
    y_train.to_csv("../data/processed/y_train.csv",    index=False)
    y_test.to_csv("data/processed/y_test.csv",      index=False)

    with open("../data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"Train : {X_train_s.shape[0]} | Test : {X_test_s.shape[0]}")
    return X_train_s, X_test_s, y_train, y_test