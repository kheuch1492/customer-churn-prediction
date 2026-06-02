# 📊 Guide de construction du Dashboard Power BI

Ce guide explique comment construire le dashboard interactif **« Prédiction du Churn Client »**
à partir des fichiers générés par le pipeline Python.

> Le livrable final est un fichier `churn_dashboard.pbix` à créer dans Power BI Desktop
> (gratuit, Windows). Les données nécessaires sont déjà prêtes dans ce dossier.

---

## 1. Fichiers sources à importer

| Fichier | Contenu | Usage |
|---|---|---|
| `churn_scored_customers.csv` | 7 043 clients + probabilité de churn + segment de risque | **Table principale** |
| `kpi_summary.csv` | KPI globaux pré-calculés | Cartes KPI (option) |
| `feature_importance.csv` | Importance des variables | Graphique d'importance |
| `model_comparison.csv` | Performance des modèles | Visuel « performance modèle » |

**Import :** `Accueil → Obtenir les données → Texte/CSV` → sélectionner chaque fichier →
vérifier que les types sont corrects (ex. `ChurnProbability` = Nombre décimal, `MonthlyCharges` = Nombre décimal).

---

## 2. Mesures DAX à créer

Dans la table `churn_scored_customers`, créer ces mesures (`Modélisation → Nouvelle mesure`) :

```DAX
Nb Clients = COUNTROWS('churn_scored_customers')

Nb Churn = CALCULATE([Nb Clients], 'churn_scored_customers'[Churn] = "Yes")

Taux de Churn = DIVIDE([Nb Churn], [Nb Clients])

Taux de Fidélisation = 1 - [Taux de Churn]

Clients Risque Élevé =
CALCULATE([Nb Clients], 'churn_scored_customers'[RiskSegment] = "Élevé")

Revenu Annuel à Risque =
SUMX(
    FILTER('churn_scored_customers', 'churn_scored_customers'[RiskSegment] = "Élevé"),
    'churn_scored_customers'[MonthlyCharges] * 12
)

Probabilité Moyenne de Churn = AVERAGE('churn_scored_customers'[ChurnProbability])
```

---

## 3. Page 1 — Vue d'ensemble (KPI)

Insérer 5 **cartes** (`Visualisations → Carte`) :

| Carte | Mesure |
|---|---|
| Nombre total de clients | `[Nb Clients]` → 7 043 |
| Taux de churn | `[Taux de Churn]` → 26,5 % |
| Clients à haut risque | `[Clients Risque Élevé]` → 1 662 |
| Revenus potentiellement perdus | `[Revenu Annuel à Risque]` → ≈ 1,51 M$ |
| Taux de fidélisation | `[Taux de Fidélisation]` → 73,5 % |

Mettre en forme : couleur rouge pour le churn/risque, vert pour la fidélisation.

---

## 4. Page 2 — Analyse du churn

| Visuel | Type | Configuration |
|---|---|---|
| **Répartition des clients à risque** | Histogramme empilé / Anneau | Axe : `RiskSegment` · Valeurs : `[Nb Clients]` |
| **Churn par contrat** | Barres groupées | Axe : `Contract` · Valeur : `[Taux de Churn]` |
| **Churn par ancienneté** | Colonnes | Axe : `TenureGroup` · Valeur : `[Taux de Churn]` |
| **Churn par méthode de paiement** | Barres horizontales | Axe : `PaymentMethod` · Valeur : `[Taux de Churn]` |
| **Coût mensuel vs churn** | Box / Histogramme | Axe : `Churn` · Valeur : `MonthlyCharges` |

---

## 5. Page 3 — Modèle & scoring

| Visuel | Type | Configuration |
|---|---|---|
| **Importance des variables** | Barres horizontales | Source : `feature_importance` (Top 15) |
| **Performance des modèles** | Barres | Source : `model_comparison` · `ROC_AUC_cv` |
| **Score de risque des clients** | Table | `customerID`, `tenure`, `Contract`, `MonthlyCharges`, `ChurnProbability`, `RiskSegment` |
| **Clients prioritaires** | Table filtrée | Filtrer `RiskSegment = "Élevé"`, trier par `ChurnProbability` décroissant |

Astuce : appliquer une **mise en forme conditionnelle** (dégradé rouge) sur la colonne
`ChurnProbability` pour repérer visuellement les clients prioritaires.

---

## 6. Filtres (segments) — à placer sur toutes les pages

Insérer des **segments** (`Visualisations → Segment`) pour :

- `gender` (Sexe)
- `Contract` (Type de contrat)
- `TenureGroup` (Ancienneté)
- `InternetService` (Service Internet)
- `PaymentMethod` (Méthode de paiement)

Utiliser `Format → Synchroniser les segments` pour les appliquer à toutes les pages.

---

## 7. Mise en page & thème

- Titre : **« Prédiction du Churn Client — Tableau de bord décisionnel »**
- Thème : `Affichage → Thèmes` → choisir un thème sobre (ex. « Innovate ») et personnaliser
  en rouge/vert (rouge = churn/risque, vert = fidèle).
- Ajouter une zone de texte « Insights & recommandations » reprenant le plan d'actions.

---

## 8. Publication

`Fichier → Enregistrer sous` → `churn_dashboard.pbix` dans ce dossier `powerbi/`.
Pour partager en ligne : `Publier → Power BI Service` (compte gratuit requis), puis exporter
des captures d'écran pour le portfolio / README GitHub.

---

### Colonnes disponibles dans `churn_scored_customers.csv`

Toutes les variables d'origine + variables enrichies :
`TenureGroup`, `NumServices`, `AvgChargesPerMonth`, `IsNewCustomer`, `IsMonthToMonth`,
`IsElectronicCheck`, **`ChurnProbability`**, **`ChurnPrediction`**, **`RiskSegment`**, **`RevenueAtRisk`**.
