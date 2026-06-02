"""
Nettoyage, préparation et feature engineering du dataset Telco Customer Churn.

Fonctions exposées :
- load_raw()         : charge le CSV brut
- clean_data(df)     : nettoyage (types, valeurs manquantes, doublons)
- add_features(df)   : création de nouvelles variables métier
- prepare()          : pipeline complet -> retourne le DataFrame prêt à l'analyse
"""
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # exécution en script direct
    import config

# Colonnes catégorielles "service" qui prennent les valeurs Yes/No/No internet service
SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_raw() -> pd.DataFrame:
    """Charge le fichier CSV brut."""
    return pd.read_csv(config.DATA_RAW)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage : conversion des types, gestion des manquants, doublons."""
    df = df.copy()

    # 1) TotalCharges contient des espaces " " pour les nouveaux clients (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 2) Les ~11 lignes manquantes correspondent à tenure = 0 -> TotalCharges = 0
    n_missing = int(df["TotalCharges"].isna().sum())
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 3) SeniorCitizen est codé 0/1 -> on le rend lisible (No/Yes) pour l'EDA
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 4) Suppression des doublons éventuels
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 5) Cible binaire 0/1 (utile pour les corrélations et le ML)
    df["ChurnFlag"] = (df[config.TARGET] == "Yes").astype(int)

    print(f"[clean] TotalCharges manquants traités : {n_missing} | doublons supprimés : {n_dup}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering : variables métier dérivées."""
    df = df.copy()

    # Tranches d'ancienneté (en mois)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, np.inf],
        labels=["0-12", "13-24", "25-48", "49-60", "60+"],
    )

    # Nombre de services additionnels souscrits (sur 6)
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)

    # Panier moyen rapporté à l'ancienneté (proxy de la valeur client)
    df["AvgChargesPerMonth"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # Indicateur "nouveau client" (forte propension au churn)
    df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)

    # Contrat sans engagement
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)

    # Paiement par chèque électronique (segment le plus churner)
    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    return df


def prepare(save: bool = True) -> pd.DataFrame:
    """Pipeline complet de préparation. Sauvegarde le dataset nettoyé si save=True."""
    df = load_raw()
    df = clean_data(df)
    df = add_features(df)
    if save:
        out = config.DATA_PROCESSED / "telco_clean.csv"
        df.to_csv(out, index=False)
        print(f"[prepare] Dataset nettoyé sauvegardé -> {out} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    return df


if __name__ == "__main__":
    prepare()
