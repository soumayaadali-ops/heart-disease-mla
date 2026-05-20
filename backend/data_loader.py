import pandas as pd

def charger_donnees(chemin="data/raw/heart.csv"):
    df = pd.read_csv(chemin)
    df = df.drop_duplicates()
    print(f"Données chargées : {df.shape[0]} patients, {df.shape[1]} colonnes")
    return df