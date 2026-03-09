import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)

from work.paths import (
    PLOTS_DIR,
    RESULTS_DIR,
    IMPORTANCE_DIR,
    XAI_DIR
)

from work.custom_logger import get_logger


logger = get_logger()



# =====================================================
# Plot basic model diagnostics
# -----------------------------------------------------
# Generates standard evaluation plots for the model:
# - ROC curve
# - Precision–Recall curve
# - Confusion matrix
# - Predicted probability distribution
#
# Output:
# results/plots/
# =====================================================


def plot_model_diagnostics(y_test, y_prob, y_pred, model_name):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.metrics import (
        roc_curve,
        roc_auc_score,
        confusion_matrix,
        precision_recall_curve,
        average_precision_score
    )

    y_test = np.asarray(y_test)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    # =========================
    # ROC CURVE
    # =========================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6,4))

    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")

    plt.legend()

    plt.savefig(
        os.path.join(PLOTS_DIR, f"roc_{model_name}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =========================
    # PRECISION RECALL CURVE
    # =========================
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    # sklearn zwraca punkty w odwrotnej kolejności
    precision = precision[::-1]
    recall = recall[::-1]

    ap = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(6,4))

    plt.plot(recall, precision, label=f"AP = {ap:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision Recall Curve - {model_name}")

    plt.legend()

    plt.savefig(
        os.path.join(PLOTS_DIR, f"precision_recall_{model_name}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(f"Confusion Matrix - {model_name}")

    plt.savefig(
        os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =========================
    # PROBABILITY HISTOGRAM
    # =========================
    plt.figure(figsize=(6,4))

    plt.hist(y_prob, bins=100)

    plt.xlabel("Predicted probability")
    plt.ylabel("Count")

    plt.title(f"Predicted Probability Distribution - {model_name}")

    plt.savefig(
        os.path.join(PLOTS_DIR, f"probability_hist_{model_name}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



# =====================================================
# Plot ranking disagreement heatmap
# -----------------------------------------------------
# Visualizes disagreement between feature rankings from
# different explanation methods:
# - permutation importance
# - SHAP importance
# - surrogate importance
#
# The heatmap shows feature ranks across methods.
#
# Output:
# results/plots/ranking_disagreement_heatmap_MODEL.png
# =====================================================
def plot_ranking_disagreement_heatmap(model_name, top_n=20):

    logger.info(f"{model_name}: plotting ranking disagreement heatmap")

    path = os.path.join(
        XAI_DIR,
        f"ranking_disagreement_{model_name}.csv"
    )

    df = pd.read_csv(path)

    # select top features based on average rank
    df["avg_rank"] = df[["perm_rank", "shap_rank", "surrogate_rank"]].mean(axis=1)

    df = df.sort_values("avg_rank")

    df_top = df.head(top_n)

    heatmap_data = df_top.set_index("feature")[
        ["perm_rank", "shap_rank", "surrogate_rank"]
    ]

    plt.figure(figsize=(8,6))

    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="coolwarm_r",
        linewidths=0.5
    )

    plt.title(f"Feature ranking disagreement ({model_name})")

    plot_path = os.path.join(
        PLOTS_DIR,
        f"ranking_disagreement_heatmap_{model_name}.png"
    )

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    logger.info(f"{model_name}: disagreement heatmap saved")



# =====================================================
# Plot SHAP vs Permutation importance scatter
# -----------------------------------------------------
# Compares feature importance values produced by two
# different explanation methods:
# - permutation importance
# - SHAP importance
#
# Each point represents a feature. The plot shows whether
# both methods assign similar importance to the same
# features or produce different rankings.
#
# Output:
# results/plots/importance_scatter_MODEL.png
# =====================================================
def plot_importance_scatter(model_name):

    logger.info(f"{model_name}: plotting SHAP vs permutation scatter")

    perm_path = os.path.join(
        IMPORTANCE_DIR,
        f"permutation_importance_{model_name}.csv"
    )

    shap_path = os.path.join(
        IMPORTANCE_DIR,
        f"shap_importance_{model_name}.csv"
    )

    perm_df = pd.read_csv(perm_path)
    shap_df = pd.read_csv(shap_path)

    df = perm_df.merge(shap_df, on="feature")

    plt.figure(figsize=(6,6))

    plt.scatter(
        df["permutation_importance"],
        df["shap_importance"],
        alpha=0.5
    )

    plt.xlabel("Permutation importance")
    plt.ylabel("SHAP importance")

    plt.title(f"SHAP vs Permutation ({model_name})")

    path = os.path.join(
        PLOTS_DIR,
        f"importance_scatter_{model_name}.png"
    )

    plt.savefig(path, bbox_inches="tight")
    plt.close()

    logger.info(f"{model_name}: importance scatter saved")



# =====================================================
# Plot attribution comparison
# -----------------------------------------------------
# Compares the top features identified by different
# explanation methods:
# - permutation importance
# - SHAP importance
#
# The plot shows how the top feature lists differ
# between the two methods.
#
# Output:
# results/plots/attribution_comparison_MODEL.png
# =====================================================
def plot_attribution_comparison(model_name):

    path = os.path.join(
        XAI_DIR,
        f"attribution_inconsistency_{model_name}.csv"
    )

    df = pd.read_csv(path)

    df_plot = df.head(15)

    plt.figure(figsize=(8,6))

    plt.plot(df_plot["perm_top30"], label="Permutation")
    plt.plot(df_plot["shap_top30"], label="SHAP")

    plt.xticks(rotation=90)

    plt.legend()

    plt.title(f"Attribution comparison ({model_name})")

    path_plot = os.path.join(
        PLOTS_DIR,
        f"attribution_comparison_{model_name}.png"
    )

    plt.savefig(path_plot, bbox_inches="tight")
    plt.close()




# =====================================================
# Plot manifold neighbors
# -----------------------------------------------------
# Visualizes local neighborhood structure used in the
# manifold stability analysis.
#
# The plot shows:
# - background data points in PCA space
# - neighbors of the selected anchor point
# - the anchor point itself
#
# This helps illustrate whether nearby points receive
# similar predictions from the model.
#
# Output:
# results/plots/manifold_neighbors_MODEL.png
# =====================================================
def plot_manifold_neighbors(model_name):

    path = os.path.join(
        XAI_DIR,
        f"manifold_points_{model_name}.parquet"
    )

    df = pd.read_parquet(path)

    plt.figure(figsize=(6,6))

    base = df[(df.is_neighbor == False) & (df.is_anchor == False)]

    plt.scatter(
        base.pca1,
        base.pca2,
        s=5,
        alpha=0.2,
        label="data"
    )

    neigh = df[df.is_neighbor]

    plt.scatter(
        neigh.pca1,
        neigh.pca2,
        color="red",
        s=30,
        label="neighbors"
    )

    anchor = df[df.is_anchor]

    plt.scatter(
        anchor.pca1,
        anchor.pca2,
        color="black",
        s=80,
        label="anchor"
    )

    plt.legend()

    plt.title(f"Manifold neighbors ({model_name})")

    path_plot = os.path.join(
        PLOTS_DIR,
        f"manifold_neighbors_{model_name}.png"
    )

    plt.savefig(path_plot, bbox_inches="tight")

    plt.close()

    logger.info(f"{model_name}: manifold plot saved")



# =====================================================
# Plot Gaussian noise stability histogram
# -----------------------------------------------------
# Visualizes prediction stability under small Gaussian
# perturbations added to the input sample.
#
# The histogram shows how often the model prediction
# remains unchanged after adding noise.
#
# High values indicate robust predictions, while
# low values suggest sensitivity to small input changes.
#
# Output:
# results/plots/gaussian_stability_hist_MODEL.png
# =====================================================
def plot_gaussian_stability_hist(model_name):

    path = os.path.join(
        XAI_DIR,
        f"gaussian_stability_{model_name}.csv"
    )

    df = pd.read_csv(path)

    plt.figure(figsize=(6,4))

    plt.hist(
        df["gaussian_prediction_stability"],
        bins=20
    )

    plt.xlabel("Prediction stability")
    plt.ylabel("Count")

    plt.title(f"Gaussian stability ({model_name})")

    plot_path = os.path.join(
        PLOTS_DIR,
        f"gaussian_stability_hist_{model_name}.png"
    )

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
