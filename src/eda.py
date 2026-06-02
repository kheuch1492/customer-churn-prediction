"""
Analyse exploratoire (EDA) : génère et sauvegarde les visualisations clés
dans reports/figures/.
"""
import matplotlib

matplotlib.use("Agg")  # backend non interactif (sauvegarde fichiers)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from . import config
    from .data_prep import prepare
except ImportError:
    import config
    from data_prep import prepare

sns.set_theme(style="whitegrid", palette="Set2")
PALETTE = {"No": "#4C9F70", "Yes": "#E15759"}


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] figure -> {path.name}")


def fig_churn_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["No", "Yes"]
    counts = df["Churn"].value_counts().reindex(order)
    sns.barplot(x=order, y=counts.values, hue=order, palette=PALETTE, ax=ax, legend=False)
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, f"{v}\n({v/total:.1%})", ha="center", fontweight="bold")
    ax.set_title("Répartition du churn client")
    ax.set_ylabel("Nombre de clients")
    ax.set_xlabel("Churn")
    _save(fig, "01_churn_distribution.png")


def fig_churn_by_contract(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(df["Contract"], df["Churn"], normalize="index")["Yes"].sort_values()
    sns.barplot(x=ct.index, y=ct.values * 100, hue=ct.index, palette="Reds", ax=ax, legend=False)
    for i, v in enumerate(ct.values):
        ax.text(i, v * 100 + 1, f"{v:.1%}", ha="center", fontweight="bold")
    ax.set_title("Taux de churn par type de contrat")
    ax.set_ylabel("Taux de churn (%)")
    ax.set_xlabel("Contrat")
    _save(fig, "02_churn_by_contract.png")


def fig_churn_by_tenure(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(df["TenureGroup"], df["Churn"], normalize="index")["Yes"]
    sns.barplot(x=ct.index, y=ct.values * 100, hue=ct.index, palette="viridis", ax=ax, legend=False)
    ax.set_title("Taux de churn par ancienneté (mois)")
    ax.set_ylabel("Taux de churn (%)")
    ax.set_xlabel("Tranche d'ancienneté")
    _save(fig, "03_churn_by_tenure.png")


def fig_churn_by_payment(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ct = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index")["Yes"].sort_values()
    sns.barplot(y=ct.index, x=ct.values * 100, hue=ct.index, palette="rocket", ax=ax, legend=False)
    for i, v in enumerate(ct.values):
        ax.text(v * 100 + 0.5, i, f"{v:.1%}", va="center", fontweight="bold")
    ax.set_title("Taux de churn par méthode de paiement")
    ax.set_xlabel("Taux de churn (%)")
    ax.set_ylabel("")
    _save(fig, "04_churn_by_payment.png")


def fig_monthly_charges_box(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette=PALETTE, ax=ax, legend=False)
    ax.set_title("Distribution du coût mensuel selon le churn")
    ax.set_ylabel("Coût mensuel ($)")
    _save(fig, "05_monthly_charges_box.png")


def fig_tenure_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, palette=PALETTE,
                 multiple="stack", ax=ax)
    ax.set_title("Distribution de l'ancienneté selon le churn")
    ax.set_xlabel("Ancienneté (mois)")
    _save(fig, "06_tenure_distribution.png")


def fig_correlation_heatmap(df):
    num = df[["tenure", "MonthlyCharges", "TotalCharges", "NumServices",
              "AvgChargesPerMonth", "ChurnFlag"]]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Heatmap de corrélation (variables numériques)")
    _save(fig, "07_correlation_heatmap.png")


def run_all():
    df = prepare(save=True)
    global pd
    import pandas as pd  # noqa: pour les crosstab ci-dessus
    fig_churn_distribution(df)
    fig_churn_by_contract(df)
    fig_churn_by_tenure(df)
    fig_churn_by_payment(df)
    fig_monthly_charges_box(df)
    fig_tenure_distribution(df)
    fig_correlation_heatmap(df)
    print("[eda] Toutes les figures EDA ont été générées.")
    return df


import pandas as pd  # import au niveau module pour les fonctions crosstab

if __name__ == "__main__":
    run_all()
