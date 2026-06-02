"""
Pipeline de Machine Learning pour la prédiction du churn.

Étapes :
1. Préparation features / cible
2. Préprocessing (OneHot + StandardScaler) via ColumnTransformer
3. Comparaison de 6 modèles avec validation croisée (SMOTE intégré)
4. Optimisation des hyperparamètres du meilleur modèle
5. Évaluation sur jeu de test (accuracy, precision, recall, F1, ROC-AUC, matrice)
6. Importance des variables
7. Sauvegarde du modèle (.pkl) + exports Power BI (scoring de tous les clients)
"""
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

try:
    from . import config
    from .data_prep import prepare
except ImportError:
    import config
    from data_prep import prepare

sns.set_theme(style="whitegrid")

# Colonnes à exclure des features (identifiant + fuites de la cible)
DROP_COLS = [config.ID_COL, config.TARGET, "ChurnFlag"]


def build_xy(df):
    """Sépare features X et cible y, et identifie colonnes num/cat."""
    X = df.drop(columns=DROP_COLS)
    y = df["ChurnFlag"]
    num_cols = X.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    """ColumnTransformer : standardisation des numériques + one-hot des catégorielles."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def get_models():
    """Dictionnaire des 6 modèles candidats."""
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=rs),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=rs, n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }


def compare_models(preprocessor, X_train, y_train):
    """Validation croisée (5 folds) avec SMOTE pour chaque modèle. Retourne un DataFrame trié."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for name, model in get_models().items():
        pipe = ImbPipeline([
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
            ("clf", model),
        ])
        auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        rec = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall", n_jobs=-1)
        rows.append({
            "Modèle": name,
            "ROC_AUC_cv": auc.mean(), "ROC_AUC_std": auc.std(),
            "F1_cv": f1.mean(), "Recall_cv": rec.mean(),
        })
        print(f"[cv] {name:20s} AUC={auc.mean():.4f}±{auc.std():.3f}  F1={f1.mean():.4f}  Recall={rec.mean():.4f}")
    res = pd.DataFrame(rows).sort_values("ROC_AUC_cv", ascending=False).reset_index(drop=True)
    res.to_csv(config.POWERBI_DIR / "model_comparison.csv", index=False)
    return res


def tune_best(best_name, preprocessor, X_train, y_train):
    """Optimisation légère des hyperparamètres du meilleur modèle via GridSearchCV."""
    rs = config.RANDOM_STATE
    base = get_models()[best_name]
    grids = {
        "Gradient Boosting": {
            "clf__n_estimators": [150, 300],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        },
        "XGBoost": {
            "clf__n_estimators": [300, 500],
            "clf__learning_rate": [0.03, 0.05],
            "clf__max_depth": [3, 4],
        },
        "Random Forest": {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [None, 12, 20],
            "clf__min_samples_leaf": [1, 3],
        },
        "Logistic Regression": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        },
    }
    grid = grids.get(best_name, {})
    pipe = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=rs)),
        ("clf", base),
    ])
    if not grid:
        pipe.fit(X_train, y_train)
        return pipe, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    print(f"[tune] meilleurs paramètres : {search.best_params_}  (AUC cv={search.best_score_:.4f})")
    return search.best_estimator_, search.best_params_


def evaluate(model, X_test, y_test):
    """Calcule toutes les métriques + sauvegarde matrice de confusion et courbe ROC."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print("\n[eval] Rapport de classification (jeu de test) :")
    print(classification_report(y_test, pred, target_names=["Fidèle", "Churn"]))

    # Matrice de confusion
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Fidèle", "Churn"], yticklabels=["Fidèle", "Churn"])
    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réel")
    fig.savefig(config.FIGURES_DIR / "08_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#E15759", lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend(loc="lower right")
    fig.savefig(config.FIGURES_DIR / "09_roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return metrics


def feature_importance(model, num_cols, cat_cols):
    """Extrait l'importance des variables (coefficients ou importances) après one-hot."""
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    feat_names = num_cols + list(
        prep.named_transformers_["cat"].get_feature_names_out(cat_cols)
    )
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return None
    fi = (pd.DataFrame({"Variable": feat_names, "Importance": imp})
          .sort_values("Importance", ascending=False).reset_index(drop=True))
    fi.to_csv(config.POWERBI_DIR / "feature_importance.csv", index=False)

    top = fi.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=top, y="Variable", x="Importance", hue="Variable", palette="mako", ax=ax, legend=False)
    ax.set_title("Top 15 des variables influençant le churn")
    fig.savefig(config.FIGURES_DIR / "10_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[fi] Importance des variables sauvegardée.")
    return fi


def risk_segment(p):
    if p >= config.RISK_HIGH:
        return "Élevé"
    if p >= config.RISK_LOW:
        return "Moyen"
    return "Faible"


def score_customers(model, df):
    """Score tous les clients et exporte le fichier pour Power BI."""
    X = df.drop(columns=DROP_COLS)
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["ChurnProbability"] = proba.round(4)
    out["ChurnPrediction"] = (proba >= 0.5).astype(int)
    out["RiskSegment"] = [risk_segment(p) for p in proba]
    # Revenu annuel potentiellement à risque (clients à risque élevé)
    out["RevenueAtRisk"] = np.where(out["RiskSegment"] == "Élevé",
                                    out["MonthlyCharges"] * 12, 0).round(2)
    path = config.POWERBI_DIR / "churn_scored_customers.csv"
    out.to_csv(path, index=False)
    print(f"[score] {len(out)} clients scorés -> {path.name}")
    return out


def main():
    print("=" * 70)
    print("PIPELINE DE PRÉDICTION DU CHURN")
    print("=" * 70)
    df = prepare(save=True)
    X, y, num_cols, cat_cols = build_xy(df)
    print(f"[data] {X.shape[0]} clients, {X.shape[1]} features ({len(num_cols)} num, {len(cat_cols)} cat)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    preprocessor = build_preprocessor(num_cols, cat_cols)

    print("\n--- 1. Comparaison des modèles (CV 5 folds + SMOTE) ---")
    results = compare_models(preprocessor, X_train, y_train)
    best_name = results.iloc[0]["Modèle"]
    print(f"\n>>> Meilleur modèle (ROC-AUC CV) : {best_name}")

    print("\n--- 2. Optimisation des hyperparamètres ---")
    best_model, best_params = tune_best(best_name, preprocessor, X_train, y_train)

    print("\n--- 3. Évaluation sur le jeu de test ---")
    metrics = evaluate(best_model, X_test, y_test)
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}")

    print("\n--- 4. Importance des variables ---")
    fi = feature_importance(best_model, num_cols, cat_cols)

    print("\n--- 5. Sauvegarde du modèle ---")
    model_path = config.MODELS_DIR / "best_churn_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"[save] Modèle sauvegardé -> {model_path}")

    print("\n--- 6. Scoring de tous les clients (export Power BI) ---")
    scored = score_customers(best_model, df)

    # KPI globaux pour le rapport et le dashboard
    n_high = int((scored["RiskSegment"] == "Élevé").sum())
    kpis = {
        "best_model": best_name,
        "best_params": best_params,
        "n_customers": int(len(df)),
        "churn_rate": float(df["ChurnFlag"].mean()),
        "retention_rate": float(1 - df["ChurnFlag"].mean()),
        "n_high_risk": n_high,
        "revenue_at_risk_annual": float(scored["RevenueAtRisk"].sum()),
        "metrics_test": metrics,
        "top_features": fi.head(10).to_dict(orient="records") if fi is not None else [],
    }
    with open(config.REPORTS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2, ensure_ascii=False)
    # Export KPI au format long pour Power BI
    pd.DataFrame([
        {"KPI": "Nombre total de clients", "Valeur": kpis["n_customers"]},
        {"KPI": "Taux de churn", "Valeur": round(kpis["churn_rate"], 4)},
        {"KPI": "Taux de fidélisation", "Valeur": round(kpis["retention_rate"], 4)},
        {"KPI": "Clients à risque élevé", "Valeur": kpis["n_high_risk"]},
        {"KPI": "Revenu annuel à risque ($)", "Valeur": round(kpis["revenue_at_risk_annual"], 2)},
        {"KPI": "ROC-AUC du modèle", "Valeur": round(metrics["roc_auc"], 4)},
    ]).to_csv(config.POWERBI_DIR / "kpi_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print(f"  Modèle retenu        : {best_name}")
    print(f"  ROC-AUC (test)       : {metrics['roc_auc']:.4f}")
    print(f"  Recall churn (test)  : {metrics['recall']:.4f}")
    print(f"  Clients à risque élevé : {n_high}")
    print(f"  Revenu annuel à risque : {kpis['revenue_at_risk_annual']:,.0f} $")
    print("=" * 70)
    return kpis


if __name__ == "__main__":
    main()
