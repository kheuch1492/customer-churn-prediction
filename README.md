# 📉 Prédiction du Churn Client — Telco

> Projet **Data Science / Business Intelligence** combinant Machine Learning et Power BI pour anticiper le départ des clients d'un opérateur télécom et orienter les actions de rétention.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green)
![PowerBI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)

🔗 **[Voir le dashboard interactif en ligne →](https://kheuch1492.github.io/customer-churn-prediction/)**

---

## 🎯 Objectif

Analyser les données de **7 043 clients** d'une entreprise télécom afin de :

1. Identifier les **facteurs** qui poussent les clients à partir.
2. **Prédire** les clients à risque élevé de churn.
3. Caractériser les **clients fidèles**.
4. **Segmenter** les profils à risque (Faible / Moyen / Élevé).
5. Proposer des **actions concrètes** de rétention.

## 📊 Résultats clés

| Indicateur | Valeur |
|---|---|
| Clients analysés | **7 043** |
| Taux de churn observé | **26,5 %** |
| Meilleur modèle | **Gradient Boosting** |
| ROC-AUC (test) | **0,844** |
| Recall churn (test) | **0,69** |
| Clients à risque élevé identifiés | **1 662** |
| Revenu annuel potentiellement à risque | **≈ 1,51 M$** |

**Top facteurs de churn :** contrat *Month-to-month* · absence de *OnlineSecurity* et *TechSupport* · faible ancienneté · fibre optique · paiement par chèque électronique.

## 🗂️ Structure du projet

```
customer-churn-prediction/
├── data/
│   ├── raw/                 # Dataset Telco original (CSV)
│   └── processed/           # Données nettoyées (telco_clean.csv)
├── notebooks/
│   └── churn_analysis.ipynb # Notebook complet (EDA → ML → conclusions)
├── src/
│   ├── config.py            # Chemins & paramètres
│   ├── data_prep.py         # Nettoyage & feature engineering
│   ├── eda.py               # Génération des visualisations
│   ├── train_model.py       # Pipeline ML complet
│   └── build_notebook.py    # Génère le notebook
├── models/
│   └── best_churn_model.pkl # Modèle entraîné (pipeline sklearn complet)
├── powerbi/
│   ├── churn_dashboard.pbix        # Dashboard Power BI finalisé (3 pages)
│   ├── churn_scored_customers.csv  # Clients + probabilité + segment de risque
│   ├── kpi_summary.csv             # KPI globaux
│   ├── model_comparison.csv        # Comparaison des modèles
│   ├── feature_importance.csv      # Importance des variables
│   ├── bg_branded_1920x1080.png    # Arrière-plan dashboard (page KPI)
│   ├── bg_subtle_1920x1080.png     # Arrière-plan dashboard (pages graphiques)
│   └── GUIDE_POWERBI.md            # Guide de construction du dashboard
├── reports/
│   ├── figures/             # Toutes les visualisations (PNG)
│   ├── metrics_summary.json # Synthèse des métriques
│   └── churn_report.pdf     # Rapport de synthèse
├── docs/
│   └── index.html           # Dashboard web interactif (Plotly, GitHub Pages)
├── requirements.txt
└── README.md
```

## 🛠️ Stack technique

**Python** · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Jupyter · **Power BI**

## 🚀 Reproduire le projet

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. (Le dataset est déjà dans data/raw/. Sinon le retélécharger.)

# 3. Lancer l'analyse exploratoire (génère les figures)
python src/eda.py

# 4. Entraîner les modèles et générer tous les livrables
python src/train_model.py

# 5. (Optionnel) Régénérer le notebook, le rapport PDF et la page web
python src/build_notebook.py
python src/build_report.py
python src/build_webpage.py
```

## 🌐 Dashboard web interactif

Une page web autonome (Plotly) est disponible dans [`docs/index.html`](docs/index.html) :
cartes KPI, filtres dynamiques (sexe, contrat, ancienneté, service internet, paiement),
graphiques de churn recalculés à la volée, importance des variables et liste des clients
prioritaires à contacter.

```bash
# Visualiser en local
python -m http.server 8077 --directory docs
# puis ouvrir http://localhost:8077
```

> Pour le publier : activer **GitHub Pages** sur le dossier `/docs` du dépôt.

## 🔬 Méthodologie

1. **Nettoyage** — conversion de `TotalCharges` (texte → numérique), gestion des 11 valeurs manquantes (clients `tenure = 0`), suppression des doublons.
2. **Feature engineering** — tranches d'ancienneté, nombre de services souscrits, panier moyen, indicateurs métier (nouveau client, contrat mensuel, chèque électronique).
3. **EDA** — distribution du churn, croisements contrat / ancienneté / paiement / coût, heatmap de corrélation.
4. **Modélisation** — comparaison de **6 algorithmes** (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, KNN) en validation croisée 5 folds, avec **SMOTE intégré au pipeline** (pas de fuite de données).
5. **Optimisation** — `GridSearchCV` sur le meilleur modèle.
6. **Évaluation** — accuracy, precision, recall, F1, ROC-AUC, matrice de confusion, courbe ROC.
7. **Interprétation** — classement des variables les plus influentes.
8. **Scoring & BI** — probabilité de churn et segment de risque pour chaque client, exportés vers Power BI.

## 📈 Comparaison des modèles (ROC-AUC en validation croisée)

| Modèle | ROC-AUC | F1 | Recall |
|---|---|---|---|
| **Gradient Boosting** | **0,846** | 0,626 | 0,658 |
| XGBoost | 0,844 | 0,618 | 0,619 |
| Logistic Regression | 0,843 | — | — |
| Random Forest | 0,824 | 0,585 | 0,564 |
| KNN | 0,813 | 0,596 | 0,825 |
| Decision Tree | — | — | — |

*(Voir `powerbi/model_comparison.csv` pour les chiffres complets.)*

## 💼 Plan d'actions business

1. **Encourager les contrats longue durée** — remises de fidélité pour migrer les clients *Month-to-month* vers des engagements 1 ou 2 ans.
2. **Vendre les services de sécurité/support** — offrir *OnlineSecurity* et *TechSupport* aux clients fibre à risque.
3. **Soigner l'onboarding** — accompagnement renforcé sur les 12 premiers mois (pic de churn).
4. **Fluidifier le paiement** — inciter à passer du chèque électronique au prélèvement automatique.
5. **Alertes automatiques** — déclencher une relance commerciale dès qu'un client dépasse un score de risque de **0,60**.

## 📦 Livrables

- ✅ Notebook Python complet ([`notebooks/churn_analysis.ipynb`](notebooks/churn_analysis.ipynb))
- ✅ Modèle ML sauvegardé ([`models/best_churn_model.pkl`](models/best_churn_model.pkl))
- ✅ Données scorées pour Power BI ([`powerbi/`](powerbi/))
- ✅ Rapport PDF de synthèse ([`reports/churn_report.pdf`](reports/churn_report.pdf))
- ✅ Dashboard Power BI interactif ([`powerbi/churn_dashboard.pbix`](powerbi/churn_dashboard.pbix)) — 3 pages (KPI, analyse du churn, modèle & clients prioritaires) avec segments synchronisés ; guide dans [`powerbi/GUIDE_POWERBI.md`](powerbi/GUIDE_POWERBI.md)

## 📚 Source de données

[Telco Customer Churn (IBM Sample Data)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7 043 clients, 21 variables.

---

*Projet réalisé dans le cadre d'un portfolio Data Analyst / BI Developer / Cheikh sall email: sall1969@outlook.fr tel 772456222