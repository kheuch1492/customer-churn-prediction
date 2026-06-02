"""
Configuration centrale du projet de prédiction du churn.
Tous les chemins sont relatifs à la racine du projet (customer-churn-prediction/).
"""
from pathlib import Path

# Racine du projet = dossier parent de /src
ROOT = Path(__file__).resolve().parents[1]

# Arborescence des données / livrables
DATA_RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
POWERBI_DIR = ROOT / "powerbi"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproductibilité
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Colonne cible
TARGET = "Churn"
ID_COL = "customerID"

# Seuils de segmentation du score de risque (probabilité de churn)
RISK_LOW = 0.30   # < 0.30  -> Faible
RISK_HIGH = 0.60  # >= 0.60 -> Élevé ; entre les deux -> Moyen

# S'assure que les dossiers de sortie existent
for _d in (DATA_PROCESSED, MODELS_DIR, POWERBI_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
