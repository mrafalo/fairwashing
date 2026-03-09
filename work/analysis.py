import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from work.paths import *
from work.custom_logger import get_logger

from work.visualization import (
    plot_attribution_comparison,
    plot_model_diagnostics,
    plot_ranking_disagreement_heatmap,
    plot_importance_scatter,
    plot_manifold_neighbors,
    plot_gaussian_stability_hist
)

logger = get_logger()



# ============================================================
# LOAD MODEL BUNDLE
# ------------------------------------------------------------
# Loads a previously trained model together with the list of
# features used during training.
#
# The function:
# - loads the model file saved during training
# - extracts the trained model and feature list
# - aligns the test dataset to the same feature order
# - creates a combined dataframe used later for analysis
#
# Why this is needed:
# Models may use a subset of features. This function ensures
# that the test dataset matches the exact feature set used
# during training before running explanations and analysis.
#
# Output:
# model        -> trained ML model
# feature_cols -> list of features used by the model
# X_model      -> aligned test dataset used for predictions
# df_analysis  -> dataframe used for XAI analyses
# ============================================================



def load_model_bundle(model_name, X_test, y_test):

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")

    bundle = joblib.load(model_path)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    X_model = X_test[feature_cols]

    df_analysis = X_model.copy()
    df_analysis["Label"] = y_test

    return model, feature_cols, X_model, df_analysis

# ============================================================
# GENERATE MODEL PREDICTIONS
# ------------------------------------------------------------
# Generates probability predictions and binary decisions for
# the test dataset using the trained model.
#
# The function:
# - computes prediction probabilities
# - converts probabilities into binary predictions
# - saves predictions to disk for later analysis
#
# Output file:
# results/predictions/predictions_MODEL.parquet
#
# The saved file contains:
# - true labels
# - predicted probability
# - predicted class
#
# These predictions are later used for diagnostics plots
# and explanation analyses.
# ============================================================


def generate_predictions(model, X_model, y_test, model_name):

    X_np = X_model.to_numpy(copy=False)

    y_prob = model.predict_proba(X_np)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred
    }, index=X_model.index)

    pred_path = os.path.join(
        PREDICTIONS_DIR,
        f"predictions_{model_name}.parquet"
    )

    pred_df.to_parquet(pred_path)

    logger.info(f"{model_name}: predictions saved")

    return pred_df



# ============================================================
# BASELINE GLOBAL EXPLANATION ANALYSIS
# ------------------------------------------------------------
# Computes two standard global explanation methods for the
# trained model: permutation importance and SHAP importance.
#
# The goal of this analysis is to identify which features are
# most important for the model predictions.
#
# The function:
# - samples the dataset to reduce computation cost
# - computes permutation importance (model performance drop
#   after feature shuffling)
# - computes SHAP importance (average absolute SHAP values)
# - saves importance rankings and visualizations
#
# Output files:
# results/importance/permutation_importance_MODEL.csv
# results/importance/shap_importance_MODEL.csv
#
# Generated plots:
# results/plots/perm_importance_MODEL.png
# results/plots/shap_importance_MODEL.png
# results/plots/shap_summary_MODEL.png
#
# Interpretation:
# The rankings show which features the model relies on most.
# Differences between permutation and SHAP rankings may
# indicate instability or disagreement between explanation
# methods.
# ============================================================



def analyze_baseline(df, model, model_name, mode="fast"):

    LABEL_COL = "Label"

    feature_cols = [c for c in df.columns if c != LABEL_COL]

    X = df[feature_cols]
    y = df[LABEL_COL]

    logger.info(f"{model_name}: baseline analysis started ({mode})")

    # =====================================
    # MODE SETTINGS
    # =====================================

    # fast -> faster but approximate explanations
    # full -> more precise but slower computation


    if mode == "fast":

        perm_sample = 20000
        shap_background = 500
        shap_sample = 1000

    else:  # full

        perm_sample = 50000
        shap_background = 1000
        shap_sample = 5000

    # =====================================
    # TRAIN TEST SPLIT
    # =====================================

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # =====================================
    # PERMUTATION IMPORTANCE
    # =====================================

    logger.info(f"{model_name}: computing permutation importance")

    X_perm = X_te.sample(
        min(perm_sample, len(X_te)),
        random_state=42
    )

    y_perm = y_te.loc[X_perm.index]

    perm = permutation_importance(
        model,
        X_perm,
        y_perm,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc" if hasattr(model, "predict_proba") else "accuracy",
        n_jobs=-1
    )

    perm_imp = perm.importances_mean

    perm_df = pd.DataFrame({
        "feature": feature_cols,
        "permutation_importance": perm_imp
    }).sort_values("permutation_importance", ascending=False)

    perm_csv = os.path.join(
        IMPORTANCE_DIR,
        f"permutation_importance_{model_name}.csv"
    )

    perm_df.to_csv(perm_csv, index=False)

    logger.info(f"{model_name}: permutation importance saved")

    # =====================================
    # PERMUTATION PLOT
    # =====================================

    plt.figure(figsize=(8,6))

    top_perm = perm_df.head(20)

    plt.barh(
        top_perm["feature"],
        top_perm["permutation_importance"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Permutation importance")
    plt.title(f"Permutation importance ({model_name})")

    perm_plot = os.path.join(
        PLOTS_DIR,
        f"perm_importance_{model_name}.png"
    )

    plt.savefig(perm_plot, bbox_inches="tight")
    plt.close()

    logger.info(f"{model_name}: permutation plot saved")

    # =====================================
    # SHAP ANALYSIS
    # =====================================

    shap_imp = None

    try:

        logger.info(f"{model_name}: computing SHAP")

        rng = np.random.default_rng(42)

        # background sample
        bg_idx = rng.choice(
            len(X_tr),
            size=min(shap_background, len(X_tr)),
            replace=False
        )

        X_bg = X_tr.iloc[bg_idx]

        # choose correct explainer
        if isinstance(model, LogisticRegression):

            explainer = shap.LinearExplainer(model, X_bg)

        else:

            explainer = shap.TreeExplainer(model, X_bg)

        # explanation sample
        ex_idx = rng.choice(
            len(X_te),
            size=min(shap_sample, len(X_te)),
            replace=False
        )

        X_explain = X_te.iloc[ex_idx]

        shap_vals = explainer.shap_values(X_explain)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_imp = np.mean(np.abs(shap_vals), axis=0)

        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "shap_importance": shap_imp
        }).sort_values("shap_importance", ascending=False)

        shap_csv = os.path.join(
            IMPORTANCE_DIR,
            f"shap_importance_{model_name}.csv"
        )

        shap_df.to_csv(shap_csv, index=False)

        logger.info(f"{model_name}: SHAP importance saved")

        # =====================================
        # SHAP BAR PLOT
        # =====================================

        plt.figure(figsize=(8,6))

        top_shap = shap_df.head(20)

        plt.barh(
            top_shap["feature"],
            top_shap["shap_importance"]
        )

        plt.gca().invert_yaxis()

        plt.xlabel("SHAP importance")
        plt.title(f"SHAP importance ({model_name})")

        shap_bar = os.path.join(
            PLOTS_DIR,
            f"shap_importance_{model_name}.png"
        )

        plt.savefig(shap_bar, bbox_inches="tight")
        plt.close()

        logger.info(f"{model_name}: SHAP bar plot saved")

        # =====================================
        # SHAP SUMMARY
        # =====================================

        plt.figure()

        shap.summary_plot(
            shap_vals,
            X_explain,
            show=False,
            max_display=20
        )

        shap_summary = os.path.join(
            PLOTS_DIR,
            f"shap_summary_{model_name}.png"
        )

        plt.savefig(shap_summary, bbox_inches="tight")
        plt.close()

        logger.info(f"{model_name}: SHAP summary plot saved")

    except Exception as e:

        logger.warning(
            f"{model_name}: SHAP skipped ({type(e).__name__}: {e})"
        )

    logger.info(f"{model_name}: baseline analysis finished")

    return perm_df, shap_imp


# ============================================================
# SHAP WATERFALL EXPLANATIONS (LOCAL EXPLANATIONS)
# ------------------------------------------------------------
# Generates SHAP waterfall plots for a few selected examples
# to explain individual model decisions.
#
# The function:
# - selects several representative cases from the test set:
#     • true positives (correct malware detection)
#     • false positives (benign predicted as malware)
#     • false negatives (missed malware)
# - computes SHAP values for these cases
# - generates waterfall plots showing how each feature
#   contributed to the final prediction.
#
# Output plots:
# results/xai/shap_waterfall_MODEL_i.png
#
# Interpretation:
# Each waterfall plot shows how individual features increase
# or decrease the model prediction relative to the baseline.
# This helps understand *why the model made a specific
# decision for a particular sample*.
#
# These plots are useful for analyzing model reasoning and
# identifying potential inconsistencies in explanations.
# ============================================================


def generate_shap_waterfall(
    model,
    X,
    y_true,
    y_pred,
    model_name
):

    logger.info(f"{model_name}: generating SHAP waterfall plots")

    df = X.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    # ====================================================
    # SELECT INTERESTING CASES
    # ====================================================

    malware = df[(df.y_true == 1) & (df.y_pred == 1)].head(2)
    false_pos = df[(df.y_true == 0) & (df.y_pred == 1)].head(2)
    false_neg = df[(df.y_true == 1) & (df.y_pred == 0)].head(2)

    cases = pd.concat([malware, false_pos, false_neg])

    X_cases = cases.drop(columns=["y_true", "y_pred"])

    logger.info(f"{model_name}: selected {len(X_cases)} cases for waterfall")

    # ====================================================
    # SHAP BACKGROUND SAMPLE
    # ====================================================

    bg = X.sample(
        n=min(1000, len(X)),
        random_state=42
    )

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, bg)
    else:
        explainer = shap.TreeExplainer(model, bg)

    shap_vals = explainer.shap_values(X_cases)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # ====================================================
    # GENERATE WATERFALL PLOTS
    # ====================================================

    for i in range(len(X_cases)):

        shap_values_single = shap_vals[i]

        base_value = explainer.expected_value

        plt.figure()

        shap.plots._waterfall.waterfall_legacy(
            base_value,
            shap_values_single,
            feature_names=X_cases.columns,
            max_display=15
        )

        path = os.path.join(
            XAI_DIR,
            f"shap_waterfall_{model_name}_{i}.png"
        )

        plt.savefig(path, bbox_inches="tight")
        plt.close()

    logger.info(f"{model_name}: SHAP waterfall plots saved")





# ============================================================
# SURROGATE MODEL RATIONALIZATION
# ------------------------------------------------------------
# Trains simple decision tree models to approximate the
# predictions of the original model.
#
# The goal is to test whether a complex model can be
# approximated by a simple interpretable surrogate.
#
# The function:
# - samples the dataset to reduce computation cost
# - generates pseudo-labels using the base model
# - trains shallow decision trees (depth 2–5)
# - measures fidelity (agreement with base model)
# - saves fidelity results and surrogate feature importance
#
# Output files:
# results/surrogate/surrogate_fidelity_MODEL.csv
# results/surrogate/surrogate_importance_all_MODEL.csv
#
# Interpretation:
# High fidelity means the complex model can be approximated
# by a simple decision tree. This suggests the model's
# behaviour may be explainable using simple rules.
# ============================================================


def surrogate_rationalization(df, base_model, model_name):

    logger.info(f"{model_name}: surrogate rationalization started")

    # ============================================
    # DATA SAMPLING (for faster computation)
    # ============================================

    df = df.sample(
        min(50000, len(df)),
        random_state=42
    )

    X = df.drop("Label", axis=1)
    y = df["Label"]

    feature_cols = X.columns

    # ============================================
    # TRAIN / TEST SPLIT
    # ============================================

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ============================================
    # GENERATE PSEUDO-LABELS FROM BASE MODEL
    # ============================================

    y_base_train = base_model.predict(X_train)
    y_base_test = base_model.predict(X_test)

    # ============================================
    # TRAIN SURROGATE MODELS (different depths)
    # ============================================

    depths = (2, 3, 4, 5)

    results = []

    best_fidelity = -1
    best_surrogate = None

    for depth in depths:

        surrogate = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=200,
            random_state=42
        )

        surrogate.fit(X_train, y_base_train)

        fidelity = accuracy_score(
            y_base_test,
            surrogate.predict(X_test)
        )

        logger.info(
            f"{model_name}: surrogate depth={depth} fidelity={fidelity:.4f}"
        )

        results.append({
            "depth": depth,
            "fidelity": fidelity
        })

        if fidelity > best_fidelity:

            best_fidelity = fidelity
            best_surrogate = surrogate

    # ============================================
    # SAVE SURROGATE FIDELITY RESULTS
    # ============================================

    results_df = pd.DataFrame(results)

    fidelity_path = os.path.join(
        SURROGATE_DIR,
        f"surrogate_fidelity_{model_name}.csv"
    )

    results_df.to_csv(fidelity_path, index=False)

    # ============================================
    # SAVE FEATURE IMPORTANCE OF BEST SURROGATE
    # ============================================

    imp = pd.DataFrame({
        "feature": feature_cols,
        "surrogate_importance": best_surrogate.feature_importances_
    }).sort_values(
        "surrogate_importance",
        ascending=False
    )

    imp_path = os.path.join(
        SURROGATE_DIR,
        f"surrogate_importance_all_{model_name}.csv"
    )

    imp.to_csv(imp_path, index=False)

    logger.info(f"{model_name}: surrogate rationalization finished")

    return results_df



# ============================================================
# FEATURE RANKING DISAGREEMENT ANALYSIS
# ------------------------------------------------------------
# Compares feature importance rankings obtained from three
# explanation methods:
#   • permutation importance
#   • SHAP importance
#   • surrogate model importance
#
# The function:
# - loads importance rankings computed earlier
# - merges them into a single table
# - computes the rank of each feature for each method
# - saves the combined ranking table
#
# Output file:
# results/xai/ranking_disagreement_MODEL.csv
#
# Interpretation:
# If different methods assign very different ranks to the same
# features, this indicates disagreement between explanation
# techniques. Such discrepancies may suggest that feature
# importance is unstable or method-dependent.
#
# The results are later visualized using a heatmap showing how
# feature rankings differ across explanation methods.
# ============================================================


def ranking_disagreement(model_name):

    logger.info(f"{model_name}: computing ranking disagreement")

    # ====================================================
    # LOAD IMPORTANCE FILES
    # ====================================================

    perm_path = os.path.join(
        IMPORTANCE_DIR,
        f"permutation_importance_{model_name}.csv"
    )

    shap_path = os.path.join(
        IMPORTANCE_DIR,
        f"shap_importance_{model_name}.csv"
    )

    surrogate_path = os.path.join(
        SURROGATE_DIR,
        f"surrogate_importance_all_{model_name}.csv"
    )

    perm_df = pd.read_csv(perm_path)
    shap_df = pd.read_csv(shap_path)
    sur_df = pd.read_csv(surrogate_path)

    # ====================================================
    # SELECT COLUMNS
    # ====================================================

    perm_df = perm_df[["feature", "permutation_importance"]]
    shap_df = shap_df[["feature", "shap_importance"]]
    sur_df = sur_df[["feature", "surrogate_importance"]]

    # ====================================================
    # MERGE IMPORTANCE
    # ====================================================

    df = perm_df.merge(shap_df, on="feature")
    df = df.merge(sur_df, on="feature")

    # ====================================================
    # COMPUTE RANKS
    # ====================================================

    df["perm_rank"] = df["permutation_importance"].rank(
        ascending=False
    )

    df["shap_rank"] = df["shap_importance"].rank(
        ascending=False
    )

    df["surrogate_rank"] = df["surrogate_importance"].rank(
        ascending=False
    )

    # ====================================================
    # SAVE RESULT
    # ====================================================

    path = os.path.join(
        XAI_DIR,
        f"ranking_disagreement_{model_name}.csv"
    )

    df.to_csv(path, index=False)

    logger.info(f"{model_name}: ranking disagreement saved")

    return df


def attribution_inconsistency(model_name):

    logger.info(f"{model_name}: attribution inconsistency")

    # =========================================
    # LOAD PERMUTATION IMPORTANCE
    # =========================================

    perm_path = os.path.join(
        IMPORTANCE_DIR,
        f"permutation_importance_{model_name}.csv"
    )

    perm_df = pd.read_csv(perm_path)

    perm_top = perm_df["feature"].head(30).tolist()

    # =========================================
    # LOAD SHAP IMPORTANCE
    # =========================================

    shap_path = os.path.join(
        IMPORTANCE_DIR,
        f"shap_importance_{model_name}.csv"
    )

    shap_df = pd.read_csv(shap_path)

    shap_top = shap_df["feature"].head(30).tolist()

    # =========================================
    # SAVE COMPARISON
    # =========================================

    df_out = pd.DataFrame({
        "perm_top30": perm_top,
        "shap_top30": shap_top
    })

    path = os.path.join(
        XAI_DIR,
        f"attribution_inconsistency_{model_name}.csv"
    )

    df_out.to_csv(path, index=False)

    logger.info(f"{model_name}: attribution inconsistency saved")

    return df_out




# ============================================================
# MANIFOLD PREDICTION STABILITY ANALYSIS
# ------------------------------------------------------------
# Tests whether the model gives consistent predictions for
# samples that are close to each other in the data manifold.
#
# The function:
# - projects data into a low-dimensional space using PCA
# - finds nearest neighbors of a selected point
# - compares predictions for the point and its neighbors
#
# Output files:
# results/xai/manifold_stability_MODEL.csv
# results/xai/manifold_points_MODEL.parquet
#
# Interpretation:
# High stability means nearby samples receive the same
# prediction. Low stability suggests the model decision may
# change for very similar inputs.
# ============================================================

def manifold_embedding(df, model, model_name):

    logger.info(f"{model_name}: manifold stability")

    # ============================================
    # PREPARE DATA
    # ============================================

    X = df.drop("Label", axis=1)

    rng = np.random.default_rng(42)

    idx = rng.choice(
        len(X),
        size=min(50000, len(X)),
        replace=False
    )

    X_bg = X.iloc[idx]

    # ============================================
    # PCA PROJECTION
    # ============================================

    pca = PCA(
        n_components=2,
        random_state=42
    )

    Z = pca.fit_transform(X_bg)

    # ============================================
    # NEAREST NEIGHBORS IN PCA SPACE
    # ============================================

    nn = NearestNeighbors(
        n_neighbors=200
    )

    nn.fit(Z)

    x = X.iloc[[0]]
    z = pca.transform(x)

    _, ids = nn.kneighbors(z)

    neighbor_idx = ids[0]

    # ============================================
    # PREDICTION STABILITY
    # ============================================

    X_neighbors = X_bg.iloc[neighbor_idx]

    base_pred = model.predict(x)[0]

    neighbor_preds = model.predict(X_neighbors)

    stability = np.mean(
        neighbor_preds == base_pred
    )

    stability_df = pd.DataFrame({
        "manifold_prediction_stability": [stability]
    })

    stability_path = os.path.join(
        XAI_DIR,
        f"manifold_stability_{model_name}.csv"
    )

    stability_df.to_csv(
        stability_path,
        index=False
    )

    # ============================================
    # SAVE PCA POINTS FOR VISUALIZATION
    # ============================================

    df_points = pd.DataFrame({
        "pca1": Z[:, 0],
        "pca2": Z[:, 1],
        "is_neighbor": False,
        "is_anchor": False
    })

    df_points.loc[
        neighbor_idx,
        "is_neighbor"
    ] = True

    anchor_df = pd.DataFrame({
        "pca1": z[:, 0],
        "pca2": z[:, 1],
        "is_neighbor": False,
        "is_anchor": True
    })

    df_points = pd.concat(
        [df_points, anchor_df],
        ignore_index=True
    )

    points_path = os.path.join(
        XAI_DIR,
        f"manifold_points_{model_name}.parquet"
    )

    df_points.to_parquet(points_path)

    logger.info(
        f"{model_name}: manifold results saved"
    )

    return stability



# ============================================================
# GAUSSIAN NOISE PREDICTION STABILITY
# ------------------------------------------------------------
# Tests whether the model prediction remains stable when small
# random Gaussian noise is added to the input features.
#
# The function:
# - selects one sample from the dataset
# - generates multiple noisy versions of that sample
# - compares predictions for noisy samples with the original
#
# Output file:
# results/xai/gaussian_stability_MODEL.csv
#
# Interpretation:
# High stability means predictions remain unchanged under
# small perturbations. Low stability indicates that small
# input changes may flip the model decision.
# ============================================================

def gaussian_noise_stability(df, model, model_name):

    logger.info(f"{model_name}: gaussian noise stability")

    # ============================================
    # PREPARE INPUT SAMPLE
    # ============================================

    X = df.drop("Label", axis=1)

    x = X.iloc[0].values

    # ============================================
    # GENERATE GAUSSIAN PERTURBATIONS
    # ============================================

    sigma = 0.01
    n_samples = 200

    noises = np.random.normal(
        0,
        sigma,
        size=(n_samples, len(x))
    )

    X_noise = x + noises

    # ============================================
    # COMPUTE PREDICTIONS
    # ============================================

    preds = model.predict(X_noise)

    base_pred = model.predict(
        x.reshape(1, -1)
    )[0]

    stability = np.mean(
        preds == base_pred
    )

    # ============================================
    # SAVE RESULT
    # ============================================

    df_out = pd.DataFrame({
        "gaussian_prediction_stability": [stability]
    })

    path = os.path.join(
        XAI_DIR,
        f"gaussian_stability_{model_name}.csv"
    )

    df_out.to_csv(
        path,
        index=False
    )

    logger.info(
        f"{model_name}: gaussian stability saved"
    )




# ============================================================
# PERMUTATION IMPORTANCE STABILITY
# ------------------------------------------------------------
# Tests whether the feature importance ranking produced by
# permutation importance is stable across multiple runs.
#
# The function:
# - samples the dataset several times
# - recomputes permutation importance in each run
# - converts importance values into feature rankings
# - stores rankings for all runs in a single table
#
# Output file:
# results/xai/importance_stability_MODEL.csv
#
# Interpretation:
# If feature ranks remain similar across runs, importance
# estimates are stable. Large rank changes indicate that the
# explanation may be sensitive to sampling or randomness.
# ============================================================

def importance_stability(df, model, model_name, n_runs=5):

    logger.info(f"{model_name}: importance stability")

    # ============================================
    # PREPARE DATA
    # ============================================

    X = df.drop("Label", axis=1)
    y = df["Label"]

    feature_cols = X.columns

    ranks = []

    # ============================================
    # REPEAT PERMUTATION IMPORTANCE
    # ============================================

    for i in range(n_runs):

        X_sample = X.sample(20000)

        y_sample = y.loc[X_sample.index]

        perm = permutation_importance(
            model,
            X_sample,
            y_sample,
            n_repeats=3,
            random_state=i,
            scoring="roc_auc",
            n_jobs=-1
        )

        imp = perm.importances_mean

        rank = pd.Series(
            imp,
            index=feature_cols
        ).rank(ascending=False)

        ranks.append(rank)

    # ============================================
    # SAVE RANKINGS
    # ============================================

    ranks_df = pd.concat(ranks, axis=1)

    ranks_df.columns = [
        f"run_{i}" for i in range(n_runs)
    ]

    path = os.path.join(
        XAI_DIR,
        f"importance_stability_{model_name}.csv"
    )

    ranks_df.to_csv(path)

    logger.info(
        f"{model_name}: importance stability saved"
    )



# ============================================================
# SHAP IMPORTANCE STABILITY
# ------------------------------------------------------------
# Tests whether the SHAP feature importance ranking is stable
# across multiple runs on different random samples of the data.
#
# The function:
# - repeatedly samples data and background sets
# - computes SHAP values for each run
# - converts SHAP importance into feature rankings
# - stores rankings from all runs in one table
#
# Output file:
# results/xai/shap_stability_MODEL.csv
#
# Interpretation:
# If feature ranks are similar across runs, SHAP explanations
# are stable. Large rank variations indicate that the
# explanation may depend strongly on sampling or randomness.
# ============================================================

def shap_importance_stability(df, model, model_name, n_runs=5):

    logger.info(f"{model_name}: SHAP importance stability")

    # ============================================
    # PREPARE DATA
    # ============================================

    X = df.drop("Label", axis=1)

    feature_cols = X.columns

    ranks = []

    # ============================================
    # REPEAT SHAP IMPORTANCE COMPUTATION
    # ============================================

    for i in range(n_runs):

        X_sample = X.sample(
            3000,
            random_state=i
        )

        bg = X.sample(
            500,
            random_state=i
        )

        # choose appropriate SHAP explainer
        if isinstance(model, LogisticRegression):

            explainer = shap.LinearExplainer(
                model,
                bg
            )

        else:

            explainer = shap.TreeExplainer(
                model,
                bg
            )

        shap_vals = explainer.shap_values(
            X_sample
        )

        if isinstance(shap_vals, list):

            shap_vals = shap_vals[1]

        shap_imp = np.mean(
            np.abs(shap_vals),
            axis=0
        )

        rank = pd.Series(
            shap_imp,
            index=feature_cols
        ).rank(ascending=False)

        ranks.append(rank)

    # ============================================
    # SAVE RANKINGS
    # ============================================

    ranks_df = pd.concat(
        ranks,
        axis=1
    )

    ranks_df.columns = [
        f"run_{i}" for i in range(n_runs)
    ]

    path = os.path.join(
        XAI_DIR,
        f"shap_stability_{model_name}.csv"
    )

    ranks_df.to_csv(path)

    logger.info(
        f"{model_name}: SHAP stability saved"
    )



# ============================================================
# FULL MODEL ANALYSIS PIPELINE
# ------------------------------------------------------------
# Runs the complete analysis workflow for a trained model.
#
# The pipeline performs:
# - prediction generation
# - model diagnostics plots
# - global explanation methods (Permutation, SHAP)
# - local explanations (SHAP waterfall)
# - surrogate model analysis
# - comparison of explanation methods
# - robustness tests for predictions and explanations
#
# Output:
# Multiple analysis files and plots saved in:
#   results/predictions/
#   results/importance/
#   results/surrogate/
#   results/xai/
#   results/plots/
#
# Interpretation:
# This pipeline produces all analyses used to evaluate
# explanation consistency, robustness, and interpretability
# of the trained model.
# ============================================================

def run_analysis_pipeline(model_name, X_test, y_test, mode="fast"):

    logger.info(f"{model_name}: starting analysis pipeline ({mode})")

    # ============================================
    # LOAD MODEL
    # ============================================

    model, feature_cols, X_model, df_analysis = load_model_bundle(
        model_name,
        X_test,
        y_test
    )

    # ============================================
    # GENERATE PREDICTIONS
    # ============================================

    pred_df = generate_predictions(
        model,
        X_model,
        y_test,
        model_name
    )

    # ============================================
    # MODEL DIAGNOSTICS
    # ============================================

    plot_model_diagnostics(
        pred_df["y_true"],
        pred_df["y_prob"],
        pred_df["y_pred"],
        model_name
    )

    # ============================================
    # GLOBAL EXPLANATION METHODS
    # ============================================

    analyze_baseline(
        df_analysis,
        model,
        model_name,
        mode
    )

    # ============================================
    # LOCAL EXPLANATIONS (SHAP WATERFALL)
    # ============================================

    generate_shap_waterfall(
        model,
        X_model,
        y_test,
        pred_df["y_pred"],
        model_name
    )

    # ============================================
    # SURROGATE MODEL ANALYSIS
    # ============================================

    surrogate_rationalization(
        df_analysis,
        model,
        model_name
    )

    # ============================================
    # EXPLANATION CONSISTENCY ANALYSES
    # ============================================

    attribution_inconsistency(model_name)

    plot_attribution_comparison(model_name)

    ranking_disagreement(model_name)

    plot_ranking_disagreement_heatmap(model_name)

    plot_importance_scatter(model_name)

    # ============================================
    # EXPLANATION ROBUSTNESS TESTS
    # ============================================

    shap_importance_stability(
        df_analysis,
        model,
        model_name
    )

    gaussian_noise_stability(
        df_analysis,
        model,
        model_name
    )

    plot_gaussian_stability_hist(
        model_name
    )

    # ============================================
    # MANIFOLD STABILITY ANALYSIS
    # ============================================

    manifold_embedding(
        df_analysis,
        model,
        model_name
    )

    plot_manifold_neighbors(
        model_name
    )

    logger.info(f"{model_name}: analysis pipeline finished")

    return True




# ============================================================
# MODEL COMPARISON SUMMARY
# ------------------------------------------------------------
# Aggregates key analysis metrics across all trained models
# to produce a comparison table.
#
# The function collects:
# - surrogate model fidelity
# - gaussian prediction stability
# - manifold prediction stability
#
# Output file:
# results/model_comparison.csv
#
# Interpretation:
# This table allows quick comparison of models in terms of
# interpretability and prediction robustness.
# ============================================================

def build_model_summary():

    logger.info("Building model comparison table")

    rows = []

    # ============================================
    # SCAN ANALYSIS RESULTS
    # ============================================

    for file in os.listdir(XAI_DIR):

        if file.startswith("manifold_stability_"):

            model = file.replace(
                "manifold_stability_",
                ""
            ).replace(".csv", "")

            # ====================================
            # MANIFOLD STABILITY
            # ====================================

            manifold_path = os.path.join(
                XAI_DIR,
                f"manifold_stability_{model}.csv"
            )

            manifold = pd.read_csv(
                manifold_path
            )["manifold_prediction_stability"].iloc[0]

            # ====================================
            # GAUSSIAN STABILITY
            # ====================================

            gaussian_path = os.path.join(
                XAI_DIR,
                f"gaussian_stability_{model}.csv"
            )

            gaussian = pd.read_csv(
                gaussian_path
            )["gaussian_prediction_stability"].iloc[0]

            # ====================================
            # SURROGATE MODEL FIDELITY
            # ====================================

            surrogate_path = os.path.join(
                SURROGATE_DIR,
                f"surrogate_fidelity_{model}.csv"
            )

            surrogate = pd.read_csv(
                surrogate_path
            )["fidelity"].max()

            rows.append({
                "model": model,
                "surrogate_fidelity": surrogate,
                "gaussian_stability": gaussian,
                "manifold_stability": manifold
            })

    # ============================================
    # BUILD SUMMARY TABLE
    # ============================================

    df = pd.DataFrame(rows)

    # sort models by surrogate fidelity
    df = df.sort_values(
        "surrogate_fidelity",
        ascending=False
    )

    path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    df.to_csv(path, index=False)

    logger.info("Model comparison table saved")

    return df
