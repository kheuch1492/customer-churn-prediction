"""
Génère le rapport PDF de synthèse `reports/churn_report.pdf` à partir des
métriques (metrics_summary.json) et des figures générées par le pipeline.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from . import config
except ImportError:
    import config

with open(config.REPORTS_DIR / "metrics_summary.json", encoding="utf-8") as f:
    K = json.load(f)

FIG = config.FIGURES_DIR
m = K["metrics_test"]


def text_page(pdf, title, lines, fontsize=11):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.5, 0.94, title, ha="center", fontsize=18, fontweight="bold", color="#2B3A55")
    y = 0.86
    for line, style in lines:
        weight = "bold" if style == "h" else "normal"
        size = 13 if style == "h" else fontsize
        color = "#E15759" if style == "h" else "#222222"
        fig.text(0.08, y, line, fontsize=size, fontweight=weight, color=color, va="top", wrap=True)
        y -= 0.045 if style == "h" else 0.032
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, title, image_files, ncols=2):
    n = len(image_files)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#2B3A55", y=0.97)
    for i, name in enumerate(image_files):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(mpimg.imread(FIG / name))
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    out = config.REPORTS_DIR / "churn_report.pdf"
    with PdfPages(out) as pdf:
        # --- Page 1 : Synthèse ---
        text_page(pdf, "Prédiction du Churn Client — Rapport de synthèse", [
            ("Contexte & objectif", "h"),
            ("Analyse de 7 043 clients d'un opérateur télécom afin d'anticiper les départs", "p"),
            ("(churn) et d'orienter les actions de rétention.", "p"),
            ("", "p"),
            ("Indicateurs clés", "h"),
            (f"• Clients analysés           : {K['n_customers']:,}", "p"),
            (f"• Taux de churn observé      : {K['churn_rate']:.1%}", "p"),
            (f"• Taux de fidélisation       : {K['retention_rate']:.1%}", "p"),
            (f"• Clients à risque élevé     : {K['n_high_risk']:,}", "p"),
            (f"• Revenu annuel à risque     : {K['revenue_at_risk_annual']:,.0f} $", "p"),
            ("", "p"),
            ("Modèle prédictif", "h"),
            (f"• Meilleur modèle            : {K['best_model']}", "p"),
            (f"• ROC-AUC (test)             : {m['roc_auc']:.3f}", "p"),
            (f"• Recall churn (test)        : {m['recall']:.3f}", "p"),
            (f"• Precision churn (test)     : {m['precision']:.3f}", "p"),
            (f"• F1-score (test)            : {m['f1']:.3f}", "p"),
            (f"• Accuracy (test)            : {m['accuracy']:.3f}", "p"),
            ("", "p"),
            ("Méthodologie", "h"),
            ("Nettoyage → feature engineering → EDA → comparaison de 6 modèles en", "p"),
            ("validation croisée avec SMOTE → optimisation → évaluation → scoring client.", "p"),
        ])

        # --- Page 2 : EDA ---
        image_page(pdf, "Analyse exploratoire", [
            "01_churn_distribution.png", "02_churn_by_contract.png",
            "03_churn_by_tenure.png", "04_churn_by_payment.png",
            "05_monthly_charges_box.png", "07_correlation_heatmap.png",
        ], ncols=2)

        # --- Page 3 : Performance du modèle ---
        image_page(pdf, "Performance du modèle & facteurs de churn", [
            "08_confusion_matrix.png", "09_roc_curve.png",
            "10_feature_importance.png",
        ], ncols=2)

        # --- Page 4 : Facteurs & plan d'actions ---
        top = K["top_features"][:6]
        feat_lines = [("Facteurs les plus influents", "h")]
        feat_lines += [(f"• {t['Variable']}  (importance {t['Importance']:.3f})", "p") for t in top]
        feat_lines += [
            ("", "p"),
            ("Profils à risque", "h"),
            ("Nouveaux clients · contrat Month-to-month · sans OnlineSecurity / TechSupport ·", "p"),
            ("fibre optique à facture élevée · paiement par chèque électronique.", "p"),
            ("", "p"),
            ("Plan d'actions business", "h"),
            ("1. Encourager les contrats longue durée (remises de fidélité).", "p"),
            ("2. Offrir les services de sécurité / support aux clients fibre à risque.", "p"),
            ("3. Renforcer l'onboarding sur les 12 premiers mois.", "p"),
            ("4. Migrer le chèque électronique vers le prélèvement automatique.", "p"),
            ("5. Déclencher des alertes dès qu'un score de risque dépasse 0,60.", "p"),
        ]
        text_page(pdf, "Insights & recommandations", feat_lines)

    print(f"Rapport PDF généré -> {out}")


if __name__ == "__main__":
    main()
