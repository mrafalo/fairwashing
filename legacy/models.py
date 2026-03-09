from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import work.custom_logger as cl
import work.globals as g
import work.data as d
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import shap

logger = cl.get_logger()


def surrogate_rationalization(df, base_model):
    # Goal: show that multiple interpretable surrogates can achieve similar fidelity to the black-box while producing different “fairness-looking” feature usage profiles.
    X = df.drop(["Label"], axis=1)
    y = df["Label"]

    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.25, random_state=g.SEED, stratify=y
    )

    y_base_train = base_model.predict(X_train)
    y_base_test = base_model.predict(X_test)
    depths = (2, 3, 4, 5)
    out = []
    for depth in depths:
        m2 = DecisionTreeClassifier(
            max_depth=depth, random_state=g.SEED, min_samples_leaf=200
        )
        m2.fit(X_train, y_base_train)
        fid = accuracy_score(y_base_test, m2.predict(X_test))
        out.append((depth, fid, m2.feature_importances_))

    for depth, fid, _ in out:
        print(f"random forest depth={depth}  fidelity={fid:.4f}")


def attribution_inconsistency(df, base_model):
    X = df.drop(["Label"], axis=1)
    y = df["Label"]
    feature_cols = [c for c in df.columns if c != "Label"]

    perm = permutation_importance(
        base_model,
        X,
        y,
        n_repeats=5,
        random_state=g.SEED,
        scoring="roc_auc" if hasattr(base_model, "predict_proba") else "accuracy",
        n_jobs=-1,
    )

    for i in perm.importances_mean.argsort()[::-1]:
        if perm.importances_mean[i] - 2 * perm.importances_std[i] > 0:
            print(
                f"{X.feature_names[i]:<8}"
                f"{perm.importances_mean[i]:.3f}"
                f" +/- {perm.importances_std[i]:.3f}"
            )

    bg = X.sample(n=2000, random_state=g.SEED)
    expl = shap.TreeExplainer(base_model, bg, feature_perturbation="interventional")
    sv = expl.shap_values(bg)
    sv = sv[1] if isinstance(sv, list) else sv
    shap_imp = np.mean(np.abs(sv), axis=0)
    shap_rank = np.argsort(-shap_imp)[:30]
    print("Top-30 by SHAP:", [feature_cols[i] for i in shap_rank])


def mainfold_embedding(df, base_model):
    X = df.drop(["Label"], axis=1)

    rng = np.random.default_rng(g.SEED)
    idx = rng.choice(len(X), size=min(50_000, len(X)), replace=False)
    X_bg = X.loc[idx]

    pca = PCA(n_components=10, random_state=g.SEED)
    Z_bg = pca.fit_transform(X_bg)

    nn = NearestNeighbors(n_neighbors=200).fit(Z_bg)

    def get_on_manifold_neighbors(x, k=200):
        z = pca.transform(x.reshape(1, -1))
        _, ids = nn.kneighbors(z, n_neighbors=k)
        return X_bg.loc[ids[0]]

    def pred_stability(m, x, Xn):
        p0 = m.predict(x.reshape(1, -1))[0]
        pn = m.predict(Xn)
        return float(np.mean(pn == p0))

    # pick a record
    x = X.loc[0].values
    Xn = get_on_manifold_neighbors(x, k=200)
    print("Prediction stability (on-manifold):", pred_stability(base_model, x, Xn))
    # TODO: SHAP stability, compare stabillity vs gausian noise


def model_random_forest(_X_train, _y_train, _X_test, _y_test):
    m1 = RandomForestClassifier(n_estimators=10)
    m1 = m1.fit(_X_train, _y_train)
    m1_predict = m1.predict(_X_test)
    auc = metrics.roc_auc_score(_y_test, m1_predict)

    return auc, m1


def model_predictor(_model_name, _model, _X_test, _y_test, _quality_measures):
    m1_predict = _model.predict(_X_test, verbose=False)

    if len(m1_predict.shape) > 1:
        m1_predict = m1_predict[:, 0]

    res = {}
    if "mse" in _quality_measures:
        res["mse"] = metrics.mean_squared_error(_y_test, m1_predict)

    if "mape" in _quality_measures:
        res["mape"] = mape_score(_y_test, m1_predict)

    if "r2" in _quality_measures:
        res["r2"] = metrics.r2_score(_y_test, m1_predict)

    if "auc" in _quality_measures:
        res["auc"] = metrics.roc_auc_score(_y_test, m1_predict)

    if "threshold" in _quality_measures:
        res["threshold"] = find_cutoff(_y_test, m1_predict)

    t = find_cutoff(_y_test, m1_predict)
    m1_predict_binary = [1 if x >= t else 0 for x in m1_predict]
    conf_matrix = np.round(metrics.confusion_matrix(_y_test, m1_predict_binary), 2)

    if "sensitivity" in _quality_measures:
        res["sensitivity"] = np.round(
            metrics.recall_score(_y_test, m1_predict_binary), 2
        )

    if "specificity" in _quality_measures:
        res["specificity"] = np.round(
            conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]), 2
        )

    if "precision" in _quality_measures:
        res["precision"] = np.round(
            metrics.precision_score(_y_test, m1_predict_binary), 2
        )

    if "f1" in _quality_measures:
        res["f1"] = np.round(metrics.f1_score(_y_test, m1_predict_binary), 2)

    fpr, tpr, thresholds = metrics.roc_curve(_y_test, m1_predict)

    roc_df = pd.DataFrame(
        {
            "False Positive Rate": fpr,
            "True Positive Rate": tpr,
            "Thresholds": thresholds,
        }
    )

    curr_date = datetime.now().strftime("%Y%m%d_%H%M")
    roc_filename = "results/models/roc_" + _model_name + "_" + curr_date + ".csv"
    res["roc_filename"] = roc_filename
    roc_df.to_csv(roc_filename, sep=";")

    return res, m1_predict
