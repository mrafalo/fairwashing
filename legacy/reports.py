mport numpy as np
import pandas as pd
import work
import work.models as m
import work.data as d
import work.globals as g
import work.custom_logger as cl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import shap

logger = cl.get_logger()


def analyze_baseline(df, m1):
    LABEL_COL = "Label"
    all_cols = list(df.columns)
    feature_cols = [c for c in all_cols if c != LABEL_COL]

    X = df[feature_cols]
    y = df[LABEL_COL]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    perm = permutation_importance(
        m1,
        X_te,
        y_te,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc" if hasattr(m1, "predict_proba") else "accuracy",
        n_jobs=-1,
    )
    perm_imp = perm.importances_mean
    perm_rank_idx = np.argsort(-perm_imp)

    print("\n[Step 1.2] Top 20 permutation importances:")
    for i in perm_rank_idx[:20]:
        print(f"{feature_cols[i]:<25} {perm_imp[i]:.6f}")

    # 1.3 SHAP (optional but very useful for fairwashing). If not installed, skip.
    shap_abs_mean = None
    try:
        # For RF, TreeExplainer works well. Use a small background sample for speed.
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X_tr), size=min(1000, len(X_tr)), replace=False)
        explainer = shap.TreeExplainer(
            m1, X_tr[bg_idx], feature_perturbation="interventional"
        )

        # Explain a subset of test for speed
        ex_idx = rng.choice(len(X_te), size=min(5000, len(X_te)), replace=False)
        shap_vals = explainer.shap_values(X_te[ex_idx])

        # shap_values format differs across versions/models:
        # - sometimes list [class0, class1]
        # - sometimes array (n, d)
        if isinstance(shap_vals, list) and len(shap_vals) >= 2:
            sv = shap_vals[1]
        else:
            sv = shap_vals

        shap_abs_mean = np.mean(np.abs(sv), axis=0)
        shap_rank_idx = np.argsort(-shap_abs_mean)

        print("\n[Step 1.3] Top 20 SHAP mean(|value|):")
        for i in shap_rank_idx[:20]:
            print(f"{feature_cols[i]:<25} {shap_abs_mean[i]:.6f}")

    except Exception as e:
        print(f"\n[Step 1.3] SHAP skipped (reason: {type(e).__name__}: {e})")
