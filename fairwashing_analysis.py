#!/usr/bin/env python
# coding: utf-8

# # CONFIG

# In[3]:


import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.base import clone

# In[7]:


import os

# =========================================
# WORKSPACE PATH
# =========================================

WORKSPACE_PATH = os.getcwd()

# =========================================
# DATA
# =========================================

DATA_DIR = os.path.join(WORKSPACE_PATH, "data")

# =========================================
# MODELS
# =========================================

MODELS_DIR = os.path.join(WORKSPACE_PATH, "models")

# =========================================
# RESULTS ROOT
# =========================================

RESULTS_DIR = os.path.join(WORKSPACE_PATH, "results")

# =========================================
# SUBFOLDERS RESULTS
# =========================================

DATA_DIR = os.path.join(WORKSPACE_PATH, "data")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
IMPORTANCE_DIR = os.path.join(RESULTS_DIR, "importance")
SURROGATE_DIR = os.path.join(RESULTS_DIR, "surrogate")
XAI_DIR = os.path.join(RESULTS_DIR, "xai")

# =========================================
# CREATE DIRECTORIES
# =========================================

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(IMPORTANCE_DIR, exist_ok=True)
os.makedirs(SURROGATE_DIR, exist_ok=True)
os.makedirs(XAI_DIR, exist_ok=True)

# =========================================
# INFO
# =========================================

print("Workspace:", WORKSPACE_PATH)
print("Data:", DATA_DIR)
print("Models:", MODELS_DIR)
print("Results:", RESULTS_DIR)
print("Plots:", PLOTS_DIR)
print("Predictions:", PREDICTIONS_DIR)
print("Importance:", IMPORTANCE_DIR)
print("Surrogate:", SURROGATE_DIR)
print("XAI:", XAI_DIR)
print("Data:", DATA_DIR)

# In[ ]:


import logging
import os

LOG_PATH = os.path.join(RESULTS_DIR, "analysis.log")

logger = logging.getLogger("model_analysis")

# tylko jeśli logger nie był jeszcze skonfigurowany
if not logger.handlers:

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s"
    )

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file
    fh = logging.FileHandler(LOG_PATH, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("Logger initialized")

# # DATA

# In[ ]:


X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
X_test  = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")

y_train = pd.read_parquet(f"{DATA_DIR}/part_9_cols.parquet")["Label"]
y_test  = pd.read_parquet(f"{DATA_DIR}/part_10_cols.parquet")["Label"]


mask_train = y_train != -1
mask_test  = y_test != -1

X_train = X_train[mask_train]
y_train = y_train[mask_train]

X_test  = X_test[mask_test]
y_test  = y_test[mask_test]


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# In[ ]:


import pandas as pd

DATA_DIR = "/dbfs/FileStore/tables/collaterals"

X_train = pd.read_parquet(f"{DATA_DIR}/part_9_scaled.parquet")
X_test  = pd.read_parquet(f"{DATA_DIR}/part_10_scaled.parquet")

y_train = pd.read_parquet(f"{DATA_DIR}/part_9_cols.parquet")["Label"]
y_test  = pd.read_parquet(f"{DATA_DIR}/part_10_cols.parquet")["Label"]

mask_train = y_train != -1
mask_test  = y_test != -1

X_train = X_train[mask_train]
y_train = y_train[mask_train]

X_test  = X_test[mask_test]
y_test  = y_test[mask_test]


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# In[8]:


import cpuinfo

info = cpuinfo.get_cpu_info()

print("CPU:", info["brand_raw"])
print("Cores:", info["count"])
print("Clock:", info["hz_advertised_friendly"])

# In[ ]:


# DATA_DIR = "/dbfs/FileStore/tables/collaterals"

# train = spark.read.parquet("/FileStore/tables/collaterals/part_8.parquet")

# feature_cols = [c for c in train.columns if c != "Label"]

# train = train.select(
#     *[train[c].cast("float").alias(c) for c in feature_cols],
#     train["Label"]
# )

# train_pd = train.toPandas()

# y_train = train_pd["Label"]
# X_train = train_pd.drop(columns=["Label"])

# X_train = X_train.astype("float32")

# X_train.to_parquet(f"{DATA_DIR}/part_9.parquet")
# y_train.to_frame(name="Label").to_parquet(f"{DATA_DIR}/part_9_cols.parquet")

# print("Train dataset saved")
# print(X_train.shape, y_train.shape)

# In[ ]:


# DATA_DIR = "/dbfs/FileStore/tables/collaterals"

# test = spark.read.parquet("/FileStore/tables/collaterals/part_7.parquet")

# feature_cols = [c for c in test.columns if c != "Label"]

# test = test.select(
#     *[test[c].cast("float").alias(c) for c in feature_cols],
#     test["Label"]
# )

# test_pd = test.toPandas()

# y_test = test_pd["Label"]
# X_test = test_pd.drop(columns=["Label"])

# X_test = X_test.astype("float32")

# X_test.to_parquet(f"{DATA_DIR}/part_10.parquet")
# y_test.to_frame(name="Label").to_parquet(f"{DATA_DIR}/part_10_cols.parquet")

# print("Test dataset saved")
# print(X_test.shape, y_test.shape)

# In[ ]:


# print(X_train.shape)
# print(y_train.shape)

# print(X_test.shape)
# print(y_test.shape)

# In[ ]:


# test.schema
# train.schema

# In[ ]:


# dbutils.fs.ls("dbfs:/FileStore/tables/collaterals/")

# In[ ]:


# os.listdir(DATA_DIR)

# In[ ]:


# display(dbutils.fs.head("dbfs:/FileStore/tables/collaterals/part_9_cols.parquet"))

# In[ ]:


# y_train = pd.read_parquet("/dbfs/FileStore/tables/collaterals/part_9_cols.parquet")

# print(y_train.head())
# print(y_train.shape)

# In[ ]:


# dbutils.fs.rm("dbfs:/FileStore/tables/collaterals/part_10_cols.parquet")
# dbutils.fs.rm("dbfs:/FileStore/tables/collaterals/part_10.parquet")
# dbutils.fs.rm("dbfs:/FileStore/tables/collaterals/part_9_cols.parquet")
# dbutils.fs.rm("dbfs:/FileStore/tables/collaterals/part_9.parquet")

# In[ ]:


# type(X_train)
# X_train.columns

# In[ ]:


# import os
# os.getcwd()

# In[ ]:


# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import joblib

# X_train = X_train.astype("float32")
# X_test  = X_test.astype("float32")

# scaler = StandardScaler()

# # fit tylko na train
# scaler.fit(X_train)



# In[ ]:


# joblib.dump(
#     scaler,
#     f"{DATA_DIR}/scaler_standard.pkl"
# )

# In[ ]:


# cols = X_train.columns

# X_train_scaled = scaler.transform(X_train).astype("float32")
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=cols)

# X_train_scaled.to_parquet(
#     f"{DATA_DIR}/part_9_scaled.parquet",
#     compression="snappy"
# )

# del X_train
# del X_train_scaled

# import gc
# gc.collect()

# print("Train usunięty z pamięci")

# In[ ]:


# cols = X_test.columns

# X_test_scaled = scaler.transform(X_test).astype("float32")

# X_test_scaled = pd.DataFrame(
#     X_test_scaled,
#     columns=cols
# )

# X_test_scaled.to_parquet(
#     f"{DATA_DIR}/part_10_scaled.parquet",
#     compression="snappy"
# )

# del X_test
# del X_test_scaled

# import gc
# gc.collect()

# print("Test usunięty z pamięci")

# # TRAINING

# In[ ]:



from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import numpy as np


def cross_validate_model(model, X, y, folds=3, seed=42):

    logger.info("Starting cross validation")

    skf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed
    )

    X_np = X.to_numpy(copy=False)
    y_np = y.to_numpy(copy=False)

    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np), start=1):

        logger.info(f"CV fold {fold}/{folds}")

        X_tr = X_np[train_idx]
        y_tr = y_np[train_idx]

        X_val = X_np[val_idx]
        y_val = y_np[val_idx]

        m = clone(model)

        logger.info("Starting CV model fit")

        m.fit(X_tr, y_tr)

        logger.info("CV model fit finished")

        y_pred = m.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_pred)

        logger.info(f"Fold {fold} AUC: {auc:.4f}")

        auc_scores.append(auc)

    cv_mean = np.mean(auc_scores)
    cv_std = np.std(auc_scores)

    logger.info(f"CV mean AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")

    return cv_mean, cv_std


# from sklearn.model_selection import StratifiedKFold
# from sklearn.base import clone
# from sklearn.metrics import roc_auc_score
# import numpy as np


# def cross_validate_model(model, X, y, folds=3, seed=42):

#     logger.info("Starting cross validation")

#     skf = StratifiedKFold(
#         n_splits=folds,
#         shuffle=True,
#         random_state=seed
#     )

#     X_np = X.to_numpy(copy=False)
#     y_np = y.to_numpy(copy=False)

#     auc_scores = []

#     for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np), start=1):

#         logger.info(f"CV fold {fold}/{folds}")

#         X_tr = X_np[train_idx]
#         y_tr = y_np[train_idx]

#         X_val = X_np[val_idx]
#         y_val = y_np[val_idx]

#         m = clone(model)

#         logger.info("Starting CV model fit")

#         m.fit(X_tr, y_tr)

#         logger.info("CV model fit finished")

#         if hasattr(m, "n_iter_"):
#             logger.info(f"Iterations: {m.n_iter_[0]}")

#         y_pred = m.predict_proba(X_val)[:, 1]

#         auc = roc_auc_score(y_val, y_pred)

#         logger.info(f"Fold {fold} AUC: {auc:.4f}")

#         auc_scores.append(auc)

#     cv_mean = np.mean(auc_scores)
#     cv_std = np.std(auc_scores)

#     logger.info(f"CV mean AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")

#     return cv_mean, cv_std


# # def cross_validate_model(model, X, y, folds=3, seed=42):

# #     logger.info("Starting cross validation")

# #     skf = StratifiedKFold(
# #         n_splits=folds,
# #         shuffle=True,
# #         random_state=seed
# #     )

# #     X_np = X.to_numpy(copy=False)
# #     y_np = y.to_numpy(copy=False)

# #     auc_scores = []

# #     for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np), start=1):

# #         logger.info(f"CV fold {fold}/{folds}")

# #         X_tr = X_np[train_idx]
# #         y_tr = y_np[train_idx]

# #         X_val = X_np[val_idx]
# #         y_val = y_np[val_idx]

# #         m = clone(model)


# #         logger.info("Starting final model fit")

# #         m.fit(X_tr, y_tr)

# #         logger.info("Final model fit finished")

# #         if hasattr(m, "n_iter_"):
# #             logger.info(f"Iterations: {m.n_iter_[0]}")

# #         y_pred = m.predict_proba(X_val)[:,1]

# #         auc = roc_auc_score(y_val, y_pred)

# #         logger.info(f"Fold {fold} AUC: {auc:.4f}")

# #         auc_scores.append(auc)

# #     cv_mean = np.mean(auc_scores)
# #     cv_std = np.std(auc_scores)

# #     logger.info(f"CV mean AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")

# #     return cv_mean, cv_std

# In[ ]:


from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):

    logger.info("Evaluating model on test set")

    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    results = {
        "auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    logger.info(
        f"Test metrics | "
        f"AUC: {results['auc']:.4f} "
        f"Accuracy: {results['accuracy']:.4f} "
        f"F1: {results['f1']:.4f}"
    )

    cm = confusion_matrix(y_test, y_pred)

    return results, y_prob, y_pred, cm

# from sklearn.metrics import (
#     roc_auc_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     accuracy_score,
#     confusion_matrix
# )

# def evaluate_model(model, X_test, y_test):

#     logger.info("Evaluating model on test set")

#     y_prob = model.predict_proba(X_test)[:,1]
#     y_pred = (y_prob >= 0.5).astype(int)

#     results = {
#         "auc": roc_auc_score(y_test, y_prob),
#         "accuracy": accuracy_score(y_test, y_pred),
#         "precision": precision_score(y_test, y_pred),
#         "recall": recall_score(y_test, y_pred),
#         "f1": f1_score(y_test, y_pred)
#     }

#     logger.info(
#         f"Test metrics | AUC: {results['auc']:.4f} "
#         f"Accuracy: {results['accuracy']:.4f} "
#         f"F1: {results['f1']:.4f}"
#     )

#     cm = confusion_matrix(y_test, y_pred)

#     return results, y_prob, y_pred, cm

# In[ ]:


import os
import joblib
import pandas as pd


def save_model_bundle(model, feature_cols, model_name, metrics):

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    bundle = {
        "model": model,
        "feature_cols": list(feature_cols),
        "metrics": metrics
    }

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")

    joblib.dump(bundle, model_path)

    logger.info(f"Model saved: {model_path}")

    metrics_df = pd.DataFrame([{
        "model": model_name,
        **metrics
    }])

    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")

    metrics_df.to_csv(
        metrics_path,
        mode="a",
        header=not os.path.exists(metrics_path),
        index=False
    )

    logger.info(f"Metrics saved: {metrics_path}")

# import os
# import joblib
# import pandas as pd
# def save_model_bundle(model, feature_cols, model_name, metrics):

#     os.makedirs(MODELS_DIR, exist_ok=True)
#     os.makedirs(RESULTS_DIR, exist_ok=True)

#     bundle = {
#         "model": model,
#         "feature_cols": list(feature_cols),
#         "metrics": metrics
#     }

#     model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")

#     joblib.dump(bundle, model_path)

#     logger.info(f"Model saved: {model_path}")

#     metrics_df = pd.DataFrame([{
#         "model": model_name,
#         **metrics
#     }])

#     metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")

#     metrics_df.to_csv(
#         metrics_path,
#         mode="a",
#         header=not os.path.exists(metrics_path),
#         index=False
#     )

#     logger.info(f"Metrics saved: {metrics_path}")


# # MODELS

# In[ ]:


from sklearn.linear_model import LogisticRegression

def train_logistic_l2(X_train, y_train, X_test, y_test):

    logger.info("Training model: log_l2")

    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        tol=1e-4,
        random_state=42
    )

    cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

    logger.info("Starting final model fit")

    model.fit(X_train, y_train)

    logger.info("Final model fit finished")

    test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    save_model_bundle(model, X_train.columns, "log_l2", results)

    logger.info("Model log_l2 training finished")

    return model

# from sklearn.linear_model import LogisticRegression

# def train_logistic_l2(X_train, y_train, X_test, y_test):

#     logger.info("Training model: log_l2")

#     model = LogisticRegression(
#         penalty="l2",
#         solver="lbfgs",
#         max_iter=1000,
#         tol=1e-4,
#         random_state=42

#     )

#     cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

#     model.fit(X_train, y_train)

#     test_metrics = evaluate_model(model, X_test, y_test)

#     results = {
#         "cv_auc_mean": cv_mean,
#         "cv_auc_std": cv_std,
#         **test_metrics
#     }

#     save_model_bundle(model, X_train.columns, "log_l2", results)

#     logger.info("Model log_l2 training finished")

#     return model

# In[ ]:


from sklearn.linear_model import LogisticRegression


def train_logistic_l1(X_train, y_train, X_test, y_test):

    logger.info("Training model: log_l1")

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.2,
        max_iter=1000,
        tol=1e-4,
        random_state=42
    )

    cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

    logger.info("Starting final model fit")

    model.fit(X_train, y_train)

    logger.info("Final model fit finished")

    test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    save_model_bundle(model, X_train.columns, "log_l1", results)

    logger.info("Model log_l1 training finished")

    return model

# from sklearn.linear_model import LogisticRegression

# def train_logistic_l1(X_train, y_train, X_test, y_test):

#     logger.info("Training model: log_l1")

#     model = LogisticRegression(
#         penalty="l1",
#         solver="saga",
#         C=0.2,
#         max_iter=1000,
#         tol=1e-4,
#         random_state=42
#     )

#     cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

#     model.fit(X_train, y_train)

#     test_metrics = evaluate_model(model, X_test, y_test)

#     results = {
#         "cv_auc_mean": cv_mean,
#         "cv_auc_std": cv_std,
#         **test_metrics
#     }

#     save_model_bundle(model, X_train.columns, "log_l1", results)

#     logger.info("Model log_l1 training finished")

#     return model

# In[ ]:


from sklearn.linear_model import LogisticRegression

def train_elastic_net(X_train, y_train, X_test, y_test):

    logger.info("Training model: elastic_net")

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.3,
        C=0.5,
        max_iter=200,
        tol=1e-3,
        random_state=42
    )

    # ==========================
    # SAMPLE FOR CV
    # ==========================

    cv_size = min(50000, len(X_train))

    X_cv = X_train.sample(n=cv_size, random_state=42)
    y_cv = y_train.loc[X_cv.index]

    logger.info(f"Elastic net CV sample size: {X_cv.shape}")

    cv_mean, cv_std = cross_validate_model(model, X_cv, y_cv)

    # ==========================
    # SAMPLE FOR FINAL FIT
    # ==========================

    final_size = min(200000, len(X_train))

    X_final = X_train.sample(n=final_size, random_state=42)
    y_final = y_train.loc[X_final.index]

    logger.info(f"Elastic net final training sample size: {X_final.shape}")

    # ==========================
    # FINAL TRAINING
    # ==========================

    logger.info("Starting final model fit")

    model.fit(X_final, y_final)

    logger.info("Final model fit finished")

    if hasattr(model, "n_iter_"):
        logger.info(f"Elastic net iterations: {model.n_iter_[0]}")

    test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    save_model_bundle(model, X_train.columns, "elastic_net", results)

    logger.info("Model elastic_net training finished")

    return model

# from sklearn.linear_model import LogisticRegression

# def train_elastic_net(X_train, y_train, X_test, y_test):

#     logger.info("Training model: elastic_net")

#     model = LogisticRegression(
#         penalty="elasticnet",
#         solver="saga",
#         l1_ratio=0.3,
#         C=0.5,
#         max_iter=200,
#         tol=1e-3,
#         random_state=42
#     )

#     # ==========================
#     # SAMPLE FOR CV
#     # ==========================

#     cv_size = min(50000, len(X_train))

#     X_cv = X_train.sample(n=cv_size, random_state=42)
#     y_cv = y_train.loc[X_cv.index]

#     logger.info(f"Elastic net CV sample size: {X_cv.shape}")

#     cv_mean, cv_std = cross_validate_model(model, X_cv, y_cv)

#     # ==========================
#     # FINAL TRAINING
#     # ==========================

#     logger.info("Starting final model fit (full dataset)")

#     model.fit(X_train, y_train)

#     logger.info("Final model fit finished")

#     if hasattr(model, "n_iter_"):
#         logger.info(f"Elastic net iterations: {model.n_iter_[0]}")

#     test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

#     results = {
#         "cv_auc_mean": cv_mean,
#         "cv_auc_std": cv_std,
#         **test_metrics
#     }

#     save_model_bundle(model, X_train.columns, "elastic_net", results)

#     logger.info("Model elastic_net training finished")

#     return model

# # from sklearn.linear_model import LogisticRegression

# # def train_elastic_net(X_train, y_train, X_test, y_test):

# #     logger.info("Training model: elastic_net")

# #     model = LogisticRegression(
# #         penalty="elasticnet",
# #         solver="saga",
# #         l1_ratio=0.3,
# #         C=0.5,
# #         max_iter=500,
# #         tol=1e-3,
# #         random_state=42
# #     )

# #     # ==========================
# #     # HARD LIMIT SAMPLE FOR CV
# #     # ==========================

# #     cv_size = min(50000, len(X_train))   # gwarantowane max 50k

# #     X_cv = X_train.sample(n=cv_size, random_state=42)
# #     y_cv = y_train.loc[X_cv.index]

# #     logger.info(f"Elastic net CV sample size: {X_cv.shape}")

# #     # CROSS VALIDATION
# #     cv_mean, cv_std = cross_validate_model(model, X_cv, y_cv)

# #     # ==========================
# #     # FINAL TRAINING FULL DATA
# #     # ==========================

# #     model.fit(X_train, y_train)

# #     if hasattr(model, "n_iter_"):
# #         logger.info(f"Elastic net iterations (final fit): {model.n_iter_[0]}")

# #     test_metrics = evaluate_model(model, X_test, y_test)

# #     results = {
# #         "cv_auc_mean": cv_mean,
# #         "cv_auc_std": cv_std,
# #         **test_metrics
# #     }

# #     save_model_bundle(model, X_train.columns, "elastic_net", results)

# #     logger.info("Model elastic_net training finished")

# #     return model

# In[ ]:



from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, X_test, y_test):

    logger.info("Training model: random_forest")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        max_features="sqrt",
        min_samples_split=10,
        min_samples_leaf=5,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

    logger.info("Starting final model fit")

    model.fit(X_train, y_train)

    logger.info("Final model fit finished")

    test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    save_model_bundle(model, X_train.columns, "random_forest", results)

    logger.info("Model random_forest training finished")

    return model

# from sklearn.ensemble import RandomForestClassifier

# def train_random_forest(X_train, y_train, X_test, y_test):

#     logger.info("Training model: random_forest")
#     model = RandomForestClassifier(
#         n_estimators=300,
#         max_depth=12,
#         max_features="sqrt",
#         min_samples_split=10,
#         min_samples_leaf=5,
#         bootstrap=True,
#         n_jobs=-1,
#         random_state=42
#     )

#     cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

#     logger.info("Final model fit finished")

#     model.fit(X_train, y_train)

#     logger.info("Final model fit finished")

#     test_metrics = evaluate_model(model, X_test, y_test)

#     results = {
#         "cv_auc_mean": cv_mean,
#         "cv_auc_std": cv_std,
#         **test_metrics
#     }

#     save_model_bundle(model, X_train.columns, "random_forest", results)

#     logger.info("Model random_forest training finished")

#     return model

# In[ ]:


pip install lightgbm

# In[ ]:



from lightgbm import LGBMClassifier
import lightgbm as lgb


def train_lightgbm(X_train, y_train, X_test, y_test):

    logger.info("Training model: lightgbm")

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True
    )

    cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

    logger.info("Starting final model fit")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    logger.info("Final model fit finished")

    test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    save_model_bundle(model, X_train.columns, "lightgbm", results)

    logger.info("Model lightgbm training finished")

    return model

# from lightgbm import LGBMClassifier


# def train_lightgbm(X_train, y_train, X_test, y_test):

#     logger.info("Training model: lightgbm")

#     model = LGBMClassifier(
#         n_estimators=1000,
#         learning_rate=0.05,
#         num_leaves=64,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#         n_jobs=-1
#     )

#     cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

#     logger.info("Starting final model fit")

#     model.fit(
#         X_train,
#         y_train,
#         eval_set=[(X_test, y_test)],
#         early_stopping_rounds=50,
#         verbose=False
#     )

#     logger.info("Final model fit finished")

#     test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

#     results = {
#         "cv_auc_mean": cv_mean,
#         "cv_auc_std": cv_std,
#         **test_metrics
#     }

#     save_model_bundle(model, X_train.columns, "lightgbm", results)

#     logger.info("Model lightgbm training finished")

#     return model


# # from lightgbm import LGBMClassifier

# # def train_lightgbm(X_train, y_train, X_test, y_test):

# #     logger.info("Training model: lightgbm")

# #     model = LGBMClassifier(
# #         n_estimators=1000,
# #         learning_rate=0.05,
# #         num_leaves=64,
# #         subsample=0.8,
# #         colsample_bytree=0.8,
# #         random_state=42,
# #         n_jobs=-1
# #     )

# #     cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

# #     model.fit(
# #         X_train,
# #         y_train,
# #         eval_set=[(X_test, y_test)],
# #         early_stopping_rounds=50,
# #         verbose=False
# #     )

# #     test_metrics = evaluate_model(model, X_test, y_test)

# #     results = {
# #         "cv_auc_mean": cv_mean,
# #         "cv_auc_std": cv_std,
# #         **test_metrics
# #     }

# #     save_model_bundle(model, X_train.columns, "lightgbm", results)

# #     logger.info("Model lightgbm training finished")

# #     return model

# In[ ]:


pip install xgboost

# In[ ]:


from xgboost import XGBClassifier


def train_xgboost(X_train, y_train, X_test, y_test):

    logger.info("Training model: xgboost")

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    logger.info("Starting cross validation")

    cv_mean, cv_std = cross_validate_model(
        model,
        X_train,
        y_train
    )

    logger.info("Starting final model fit")

    model.fit(
        X_train,
        y_train
    )

    logger.info("Final model fit finished")

    test_metrics, y_prob, y_pred, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    save_model_bundle(
        model,
        X_train.columns,
        "xgboost",
        results
    )

    logger.info("Model xgboost training finished")

    return model


# from xgboost import XGBClassifier
# from xgboost.callback import EarlyStopping


# def train_xgboost(X_train, y_train, X_test, y_test):

#     logger.info("Training model: xgboost")

#     model = XGBClassifier(
#         n_estimators=1000,
#         max_depth=6,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         tree_method="hist",
#         eval_metric="logloss",
#         random_state=42,
#         n_jobs=-1
#     )

#     cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

#     logger.info("Starting final model fit")

#     model.fit(
#         X_train,
#         y_train,
#         eval_set=[(X_test, y_test)],
#         callbacks=[EarlyStopping(rounds=50)],
#         verbose=False
#     )

#     logger.info("Final model fit finished")

#     test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

#     results = {
#         "cv_auc_mean": cv_mean,
#         "cv_auc_std": cv_std,
#         **test_metrics
#     }

#     save_model_bundle(model, X_train.columns, "xgboost", results)

#     logger.info("Model xgboost training finished")

#     return model

# # from xgboost import XGBClassifier


# # def train_xgboost(X_train, y_train, X_test, y_test):

# #     logger.info("Training model: xgboost")

# #     model = XGBClassifier(
# #         n_estimators=1000,
# #         max_depth=6,
# #         learning_rate=0.05,
# #         subsample=0.8,
# #         colsample_bytree=0.8,
# #         tree_method="hist",
# #         eval_metric="logloss",
# #         random_state=42,
# #         n_jobs=-1
# #     )

# #     cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

# #     logger.info("Starting final model fit")

# #     model.fit(
# #         X_train,
# #         y_train,
# #         eval_set=[(X_test, y_test)],
# #         early_stopping_rounds=50,
# #         verbose=False
# #     )

# #     logger.info("Final model fit finished")

# #     test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

# #     results = {
# #         "cv_auc_mean": cv_mean,
# #         "cv_auc_std": cv_std,
# #         **test_metrics
# #     }

# #     save_model_bundle(model, X_train.columns, "xgboost", results)

# #     logger.info("Model xgboost training finished")

# #     return model

# # # from xgboost import XGBClassifier

# # # def train_xgboost(X_train, y_train, X_test, y_test):

# # #     logger.info("Training model: xgboost")

# # #     model = XGBClassifier(
# # #         n_estimators=1000,
# # #         max_depth=6,
# # #         learning_rate=0.05,
# # #         subsample=0.8,
# # #         colsample_bytree=0.8,
# # #         tree_method="hist",
# # #         eval_metric="logloss",
# # #         random_state=42,
# # #         n_jobs=-1
# # #     )

# # #     cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

# # #     model.fit(
# # #         X_train,
# # #         y_train,
# # #         eval_set=[(X_test, y_test)],
# # #         early_stopping_rounds=50,
# # #         verbose=False
# # #     )

# # #     test_metrics = evaluate_model(model, X_test, y_test)

# # #     results = {
# # #         "cv_auc_mean": cv_mean,
# # #         "cv_auc_std": cv_std,
# # #         **test_metrics
# # #     }

# # #     save_model_bundle(model, X_train.columns, "xgboost", results)

# # #     logger.info("Model xgboost training finished")

# # #     return model

# In[ ]:


from sklearn.neural_network import MLPClassifier


def train_mlp(X_train, y_train, X_test, y_test):

    logger.info("Training model: mlp")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        batch_size=512,
        alpha=0.0005,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42
    )

    cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

    logger.info("Starting final model fit")

    model.fit(X_train, y_train)

    logger.info("Final model fit finished")

    test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    save_model_bundle(model, X_train.columns, "mlp", results)

    logger.info("Model mlp training finished")

    return model

# from sklearn.neural_network import MLPClassifier

# def train_mlp(X_train, y_train, X_test, y_test):

#     logger.info("Training model: mlp")

#     model = MLPClassifier(
#             hidden_layer_sizes=(128,64),
#             activation="relu",
#             solver="adam",
#             batch_size=512,
#             alpha=0.0005,
#             max_iter=100,
#             early_stopping=True,
#             validation_fraction=0.1,
#             n_iter_no_change=5,
#             random_state=42
#         )

#     cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

#     model.fit(X_train, y_train)

#     test_metrics = evaluate_model(model, X_test, y_test)

#     results = {
#         "cv_auc_mean": cv_mean,
#         "cv_auc_std": cv_std,
#         **test_metrics
#     }

#     save_model_bundle(model, X_train.columns, "mlp", results)

#     logger.info("Model mlp training finished")

#     return model

# In[ ]:


from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def train_svm(X_train, y_train, X_test, y_test):

    logger.info("Training model: linear_svm")

    base = LinearSVC(
        C=1.0,
        max_iter=5000,
        tol=1e-4,
        random_state=42
    )

    model = CalibratedClassifierCV(
        base,
        method="sigmoid",
        cv=3
    )

    cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

    logger.info("Starting final model fit")

    model.fit(X_train, y_train)

    logger.info("Final model fit finished")

    test_metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    save_model_bundle(model, X_train.columns, "linear_svm", results)

    logger.info("Model linear_svm training finished")

    return model


# from sklearn.svm import LinearSVC
# from sklearn.calibration import CalibratedClassifierCV

# def train_svm(X_train, y_train, X_test, y_test):

#    logger.info("Training model: linear_svm")

#     base = LinearSVC(
#         C=1.0,
#         max_iter=5000,
#         tol=1e-4,
#         random_state=42
#     )

#     model = CalibratedClassifierCV(
#         base,
#         method="sigmoid",
#         cv=3
#     )

#     cv_mean, cv_std = cross_validate_model(model, X_train, y_train)

#     model.fit(X_train, y_train)

#     test_metrics = evaluate_model(model, X_test, y_test)

#     results = {
#         "cv_auc_mean": cv_mean,
#         "cv_auc_std": cv_std,
#         **test_metrics
#     }

#     save_model_bundle(model, X_train.columns, "linear_svm", results)

#     logger.info("Model linear_svm training finished")

#     return model

# # ANALYSIS

# In[ ]:


def load_model_bundle(model_name, X_test, y_test):

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")

    bundle = joblib.load(model_path)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    X_model = X_test[feature_cols]

    df_analysis = X_model.copy()
    df_analysis["Label"] = y_test

    return model, feature_cols, X_model, df_analysis

# In[ ]:


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

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)


def plot_model_diagnostics(y_test, y_prob, y_pred, model_name):

    # =========================
    # ROC
    # =========================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()

    plt.savefig(os.path.join(PLOTS_DIR, f"roc_{model_name}.png"), dpi=300)
    plt.close()

    # =========================
    # Precision Recall
    # =========================
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision Recall Curve - {model_name}")

    plt.savefig(os.path.join(PLOTS_DIR, f"precision_recall_{model_name}.png"), dpi=300)
    plt.close()

    # =========================
    # Confusion Matrix
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")

    plt.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png"), dpi=300)
    plt.close()

    # =========================
    # Histogram
    # =========================
    plt.figure()
    plt.hist(y_prob, bins=50)

    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(f"Predicted Probability Distribution - {model_name}")

    plt.savefig(os.path.join(PLOTS_DIR, f"probability_hist_{model_name}.png"), dpi=300)
    plt.close()

# In[ ]:


# from sklearn import metrics
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import numpy as np
# import pandas as pd
# import work.custom_logger as cl
# import work.globals as g
# import work.data as d
# from datetime import datetime
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.inspection import permutation_importance
# from sklearn.neighbors import NearestNeighbors
# from sklearn.decomposition import PCA
# import shap

# logger = cl.get_logger()


# def surrogate_rationalization(df, base_model):
#     # Goal: show that multiple interpretable surrogates can achieve similar fidelity to the black-box while producing different “fairness-looking” feature usage profiles.
#     X = df.drop(["Label"], axis=1)
#     y = df["Label"]

#     X_train, X_test, _, _ = train_test_split(
#         X, y, test_size=0.25, random_state=g.SEED, stratify=y
#     )

#     y_base_train = base_model.predict(X_train)
#     y_base_test = base_model.predict(X_test)
#     depths = (2, 3, 4, 5)
#     out = []
#     for depth in depths:
#         m2 = DecisionTreeClassifier(
#             max_depth=depth, random_state=g.SEED, min_samples_leaf=200
#         )
#         m2.fit(X_train, y_base_train)
#         fid = accuracy_score(y_base_test, m2.predict(X_test))
#         out.append((depth, fid, m2.feature_importances_))

#     for depth, fid, _ in out:
#         print(f"random forest depth={depth}  fidelity={fid:.4f}")


# def attribution_inconsistency(df, base_model):
#     X = df.drop(["Label"], axis=1)
#     y = df["Label"]
#     feature_cols = [c for c in df.columns if c != "Label"]

#     perm = permutation_importance(
#         base_model,
#         X,
#         y,
#         n_repeats=5,
#         random_state=g.SEED,
#         scoring="roc_auc" if hasattr(base_model, "predict_proba") else "accuracy",
#         n_jobs=-1,
#     )

#     for i in perm.importances_mean.argsort()[::-1]:
#         if perm.importances_mean[i] - 2 * perm.importances_std[i] > 0:
#             print(
#                 f"{X.feature_names[i]:<8}"
#                 f"{perm.importances_mean[i]:.3f}"
#                 f" +/- {perm.importances_std[i]:.3f}"
#             )

#     bg = X.sample(n=2000, random_state=g.SEED)
#     expl = shap.TreeExplainer(base_model, bg, feature_perturbation="interventional")
#     sv = expl.shap_values(bg)
#     sv = sv[1] if isinstance(sv, list) else sv
#     shap_imp = np.mean(np.abs(sv), axis=0)
#     shap_rank = np.argsort(-shap_imp)[:30]
#     print("Top-30 by SHAP:", [feature_cols[i] for i in shap_rank])


# def manifold_embedding(df, base_model):
#     X = df.drop(["Label"], axis=1)

#     rng = np.random.default_rng(g.SEED)
#     idx = rng.choice(len(X), size=min(50_000, len(X)), replace=False)
#     X_bg = X.loc[idx]

#     pca = PCA(n_components=10, random_state=g.SEED)
#     Z_bg = pca.fit_transform(X_bg)

#     nn = NearestNeighbors(n_neighbors=200).fit(Z_bg)

#     def get_on_manifold_neighbors(x, k=200):
#         z = pca.transform(x.reshape(1, -1))
#         _, ids = nn.kneighbors(z, n_neighbors=k)
#         return X_bg.loc[ids[0]]

#     def pred_stability(m, x, Xn):
#         p0 = m.predict(x.reshape(1, -1))[0]
#         pn = m.predict(Xn)
#         return float(np.mean(pn == p0))

#     # pick a record
#     x = X.loc[0].values
#     Xn = get_on_manifold_neighbors(x, k=200)
#     print("Prediction stability (on-manifold):", pred_stability(base_model, x, Xn))
#     # TODO: SHAP stability, compare stabillity vs gausian noise


# # def model_random_forest(_X_train, _y_train, _X_test, _y_test):
# #     m1 = RandomForestClassifier(n_estimators=10)
# #     m1 = m1.fit(_X_train, _y_train)
# #     m1_predict = m1.predict(_X_test)
# #     auc = metrics.roc_auc_score(_y_test, m1_predict)

# #     return auc, m1


# def model_predictor(_model_name, _model, _X_test, _y_test, _quality_measures):
#     m1_predict = _model.predict(_X_test, verbose=False)

#     if len(m1_predict.shape) > 1:
#         m1_predict = m1_predict[:, 0]

#     res = {}
#     if "mse" in _quality_measures:
#         res["mse"] = metrics.mean_squared_error(_y_test, m1_predict)

#     if "mape" in _quality_measures:
#         res["mape"] = mape_score(_y_test, m1_predict)

#     if "r2" in _quality_measures:
#         res["r2"] = metrics.r2_score(_y_test, m1_predict)

#     if "auc" in _quality_measures:
#         res["auc"] = metrics.roc_auc_score(_y_test, m1_predict)

#     if "threshold" in _quality_measures:
#         res["threshold"] = find_cutoff(_y_test, m1_predict)

#     t = find_cutoff(_y_test, m1_predict)
#     m1_predict_binary = [1 if x >= t else 0 for x in m1_predict]
#     conf_matrix = np.round(metrics.confusion_matrix(_y_test, m1_predict_binary), 2)

#     if "sensitivity" in _quality_measures:
#         res["sensitivity"] = np.round(
#             metrics.recall_score(_y_test, m1_predict_binary), 2
#         )

#     if "specificity" in _quality_measures:
#         res["specificity"] = np.round(
#             conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]), 2
#         )

#     if "precision" in _quality_measures:
#         res["precision"] = np.round(
#             metrics.precision_score(_y_test, m1_predict_binary), 2
#         )

#     if "f1" in _quality_measures:
#         res["f1"] = np.round(metrics.f1_score(_y_test, m1_predict_binary), 2)

#     fpr, tpr, thresholds = metrics.roc_curve(_y_test, m1_predict)

#     roc_df = pd.DataFrame(
#         {
#             "False Positive Rate": fpr,
#             "True Positive Rate": tpr,
#             "Thresholds": thresholds,
#         }
#     )

#     curr_date = datetime.now().strftime("%Y%m%d_%H%M")
#     roc_filename = "results/models/roc_" + _model_name + "_" + curr_date + ".csv"
#     res["roc_filename"] = roc_filename
#     roc_df.to_csv(roc_filename, sep=";")

#     return res, m1_predict

# In[ ]:


%pip install shap

# In[ ]:


import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def analyze_baseline(df, model, model_name, mode="fast"):

    LABEL_COL = "Label"

    feature_cols = [c for c in df.columns if c != LABEL_COL]

    X = df[feature_cols]
    y = df[LABEL_COL]

    logger.info(f"{model_name}: baseline analysis started ({mode})")

    # =====================================
    # MODE SETTINGS
    # =====================================

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

# In[ ]:


import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


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

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os


def surrogate_rationalization(df, base_model, model_name):

    logger.info(f"{model_name}: surrogate rationalization started")

    # sampling żeby przyspieszyć
    df = df.sample(
        min(50000, len(df)),
        random_state=42
    )

    X = df.drop("Label", axis=1)
    y = df["Label"]

    feature_cols = X.columns

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # pseudo-labels z modelu bazowego
    y_base_train = base_model.predict(X_train)
    y_base_test = base_model.predict(X_test)

    depths = (2, 3, 4, 5)

    results = []

    best_fidelity = -1
    best_surrogate = None

    for depth in depths:

        surrogate = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42,
            min_samples_leaf=200
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

        # zapamiętujemy najlepszy surrogate
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_surrogate = surrogate

    # zapis fidelity
    results_df = pd.DataFrame(results)

    fidelity_path = os.path.join(
        SURROGATE_DIR,
        f"surrogate_fidelity_{model_name}.csv"
    )

    results_df.to_csv(fidelity_path, index=False)

    # zapis importance najlepszego surrogate
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

# In[ ]:


import pandas as pd
import os


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

# In[ ]:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_ranking_disagreement_heatmap(model_name, top_n=20):

    logger.info(f"{model_name}: plotting ranking disagreement heatmap")

    path = os.path.join(
        XAI_DIR,
        f"ranking_disagreement_{model_name}.csv"
    )

    df = pd.read_csv(path)

    # select top features based on average rank
    df["avg_rank"] = df[
        ["perm_rank", "shap_rank", "surrogate_rank"]
    ].mean(axis=1)

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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


def plot_attribution_comparison(model_name):

    path = os.path.join(
        RESULTS_DIR,
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

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import os

def manifold_embedding(df, model, model_name):

    logger.info(f"{model_name}: manifold stability")

    X = df.drop("Label", axis=1)

    rng = np.random.default_rng(42)

    idx = rng.choice(len(X), size=min(50000, len(X)), replace=False)

    X_bg = X.iloc[idx]

    # PCA
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_bg)

    # neighbors
    nn = NearestNeighbors(n_neighbors=200)
    nn.fit(Z)

    x = X.iloc[[0]]
    z = pca.transform(x)

    _, ids = nn.kneighbors(z)
    neighbor_idx = ids[0]

    # stability
    X_neighbors = X_bg.iloc[neighbor_idx]

    base_pred = model.predict(x)[0]
    neighbor_preds = model.predict(X_neighbors)

    stability = np.mean(neighbor_preds == base_pred)

    stability_df = pd.DataFrame({
        "manifold_prediction_stability":[stability]
    })

    stability_path = os.path.join(
        XAI_DIR,
        f"manifold_stability_{model_name}.csv"
    )

    stability_df.to_csv(stability_path, index=False)

    # punkty PCA do wykresu
    df_points = pd.DataFrame({
        "pca1": Z[:,0],
        "pca2": Z[:,1],
        "is_neighbor": False,
        "is_anchor": False
    })

    df_points.loc[neighbor_idx, "is_neighbor"] = True

    anchor_df = pd.DataFrame({
        "pca1": z[:,0],
        "pca2": z[:,1],
        "is_neighbor": False,
        "is_anchor": True
    })

    df_points = pd.concat([df_points, anchor_df], ignore_index=True)

    points_path = os.path.join(
        XAI_DIR,
        f"manifold_points_{model_name}.parquet"
    )

    df_points.to_parquet(points_path)

    logger.info(f"{model_name}: manifold results saved")

    return stability

# In[ ]:


import matplotlib.pyplot as plt

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

# In[ ]:


def gaussian_noise_stability(df, model, model_name):

    logger.info(f"{model_name}: gaussian noise stability")

    X = df.drop("Label", axis=1)

    x = X.iloc[0].values

    sigma = 0.01

    n_samples = 200

    noises = np.random.normal(
        0,
        sigma,
        size=(n_samples, len(x))
    )

    X_noise = x + noises

    preds = model.predict(X_noise)

    base_pred = model.predict(x.reshape(1,-1))[0]

    stability = np.mean(preds == base_pred)

    df_out = pd.DataFrame({
        "gaussian_prediction_stability":[stability]
    })

    path = os.path.join(
        XAI_DIR,
        f"gaussian_stability_{model_name}.csv"
    )

    df_out.to_csv(path, index=False)

    logger.info(f"{model_name}: gaussian stability saved")

# In[ ]:


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

# In[ ]:


def importance_stability(df, model, model_name, n_runs=5):

    logger.info(f"{model_name}: importance stability")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    feature_cols = X.columns

    ranks = []

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

        rank = pd.Series(imp, index=feature_cols).rank(ascending=False)

        ranks.append(rank)

    ranks_df = pd.concat(ranks, axis=1)

    ranks_df.columns = [f"run_{i}" for i in range(n_runs)]

    path = os.path.join(
        XAI_DIR,
        f"importance_stability_{model_name}.csv"
    )

    ranks_df.to_csv(path)

    logger.info(f"{model_name}: importance stability saved")

# In[ ]:


def shap_importance_stability(df, model, model_name, n_runs=5):

    logger.info(f"{model_name}: SHAP importance stability")

    X = df.drop("Label", axis=1)

    feature_cols = X.columns

    ranks = []

    for i in range(n_runs):

        X_sample = X.sample(3000, random_state=i)

        bg = X.sample(500, random_state=i)

        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, bg)
        else:
            explainer = shap.TreeExplainer(model, bg)

        shap_vals = explainer.shap_values(X_sample)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_imp = np.mean(np.abs(shap_vals), axis=0)

        rank = pd.Series(shap_imp, index=feature_cols).rank(
            ascending=False
        )

        ranks.append(rank)

    ranks_df = pd.concat(ranks, axis=1)

    ranks_df.columns = [f"run_{i}" for i in range(n_runs)]

    path = os.path.join(
        XAI_DIR,
        f"shap_stability_{model_name}.csv"
    )

    ranks_df.to_csv(path)

    logger.info(f"{model_name}: SHAP stability saved")

# # FUNCTION CALLS

# In[ ]:


# model = train_logistic_l2(X_train, y_train, X_test, y_test)
# model = train_logistic_l1(X_train, y_train, X_test, y_test)
# model = train_elastic_net(X_train, y_train, X_test, y_test)
# model = train_random_forest(X_train, y_train, X_test, y_test)
# model = train_lightgbm(X_train, y_train, X_test, y_test)
# model = train_xgboost(X_train, y_train, X_test, y_test)
# model = train_mlp(X_train, y_train, X_test, y_test)
model = train_svm(X_train, y_train, X_test, y_test)

# In[ ]:


print(X_train.shape)

# In[ ]:


# def run_analysis_pipeline(model_name, X_test, y_test, mode="fast"):

#     logger.info(f"{model_name}: starting analysis pipeline ({mode})")

#     # ============================================
#     # LOAD MODEL
#     # ============================================

#     model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")

#     bundle = joblib.load(model_path)
#     model = bundle["model"]
#     feature_cols = bundle["feature_cols"]
#     X_model = X_test[feature_cols]
#     df_analysis = X_model.copy()
#     df_analysis["Label"] = y_test

#     # ============================================
#     # PREDICTIONS
#     # ============================================

#     pred_df = generate_predictions(model, X_model, y_test, model_name)

#     # ============================================
#     # MODEL DIAGNOSTICS
#     # ============================================

#     plot_model_diagnostics(
#         pred_df["y_true"],
#         pred_df["y_prob"],
#         pred_df["y_pred"],
#         model_name
#     )

#     # ============================================
#     # BASELINE GLOBAL EXPLANATIONS
#     # ============================================

#     analyze_baseline(
#         df_analysis,
#         model,
#         model_name,
#         mode
#     )

#     # ============================================
#     # LOCAL SHAP EXPLANATIONS
#     # ============================================

#     generate_shap_waterfall(
#         model,
#         X_model,
#         y_test,
#         pred_df["y_pred"],
#         model_name
#     )

#     # ============================================
#     # SURROGATE MODEL ANALYSIS
#     # ============================================

#     surrogate_rationalization(
#         df_analysis,
#         model,
#         model_name
#     )

#     # ============================================
#     # METHOD DISAGREEMENT
#     # ============================================

#     ranking_disagreement(
#         model_name
#     )

#     plot_ranking_disagreement_heatmap(
#         model_name
#     )

#     plot_importance_scatter(
#         model_name
#     )

#     # ============================================
#     # ADDITIONAL ANALYSES
#     # ============================================

#     attribution_inconsistency(
#         df_analysis,
#         model,
#         model_name
#     )

#     plot_attribution_comparison(model_name)

#     manifold_embedding(df_analysis, model, model_name)

#     plot_manifold_neighbors(model_name)

#     shap_importance_stability(df_analysis, model, model_name)

#     gaussian_noise_stability(df_analysis, model, model_name)

#     plot_gaussian_stability_hist("log_l2")

#     logger.info(f"{model_name}: analysis pipeline finished")

# In[ ]:


import os
import joblib


def run_analysis_pipeline(model_name, X_test, y_test, mode="fast"):

    logger.info(f"{model_name}: starting analysis pipeline ({mode})")

    # ============================================
    # LOAD MODEL
    # ============================================

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")

    bundle = joblib.load(model_path)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    X_model = X_test[feature_cols]

    df_analysis = X_model.copy()
    df_analysis["Label"] = y_test

    # ============================================
    # PREDICTIONS
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
    # BASELINE GLOBAL EXPLANATIONS
    # ============================================

    analyze_baseline(
        df_analysis,
        model,
        model_name,
        mode
    )

    # ============================================
    # LOCAL SHAP EXPLANATIONS
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

    surrogate_rationalization(df_analysis, model, model_name)

    # ============================================
    # ADDITIONAL EXPLANATION ANALYSIS
    # ============================================

    attribution_inconsistency(model_name)

    # ============================================
    # METHOD DISAGREEMENT
    # ============================================

    ranking_disagreement(model_name)

    plot_ranking_disagreement_heatmap(model_name)

    plot_importance_scatter(model_name)

    # ============================================
    # ROBUSTNESS ANALYSES
    # ============================================

    shap_importance_stability(df_analysis, model, model_name)

    gaussian_noise_stability(
        df_analysis,
        model,
        model_name
    )

    plot_gaussian_stability_hist(
        model_name
    )

    # ============================================
    # MANIFOLD ANALYSIS
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

# In[ ]:


results = run_analysis_pipeline(
    "log_l2",
    X_test,
    y_test,
    mode="fast"
)

# In[ ]:


# run_analysis_pipeline(
#     "log_l1",
#     X_test,
#     y_test,
#     mode="full"
# )

# In[ ]:


# def check_pipeline_outputs(model_name):

#     files_expected = {

#         "predictions": [
#             os.path.join(PREDICTIONS_DIR, f"predictions_{model_name}.parquet")
#         ],

#         "importance": [
#             os.path.join(IMPORTANCE_DIR, f"permutation_importance_{model_name}.csv"),
#             os.path.join(IMPORTANCE_DIR, f"shap_importance_{model_name}.csv")
#         ],

#         "plots": [
#             os.path.join(PLOTS_DIR, f"perm_importance_{model_name}.png"),
#             os.path.join(PLOTS_DIR, f"shap_importance_{model_name}.png"),
#             os.path.join(PLOTS_DIR, f"shap_summary_{model_name}.png"),
#             os.path.join(PLOTS_DIR, f"importance_scatter_{model_name}.png"),
#             os.path.join(PLOTS_DIR, f"ranking_disagreement_heatmap_{model_name}.png"),
#             os.path.join(PLOTS_DIR, f"surrogate_fidelity_{model_name}.png")
#         ],

#         "surrogate": [
#             os.path.join(SURROGATE_DIR, f"surrogate_fidelity_{model_name}.csv"),
#             os.path.join(SURROGATE_DIR, f"surrogate_importance_all_{model_name}.csv")
#         ],

#         "analysis": [
#             os.path.join(RESULTS_DIR, f"ranking_disagreement_{model_name}.csv"),
#             os.path.join(RESULTS_DIR, f"attribution_inconsistency_{model_name}.csv"),
#             os.path.join(RESULTS_DIR, f"manifold_stability_{model_name}.csv"),
#             os.path.join(RESULTS_DIR, f"shap_stability_{model_name}.csv")
#         ]
#     }

#     print(f"\nChecking outputs for model: {model_name}\n")

#     for group, files in files_expected.items():

#         print(f"--- {group.upper()} ---")

#         for f in files:

#             if os.path.exists(f):
#                 print("OK   ", os.path.basename(f))
#             else:
#                 print("MISS ", os.path.basename(f))

#         print()

# In[ ]:


# plot_gaussian_stability_hist("log_l2")

# In[ ]:


# plot_attribution_comparison("log_l2")

# In[ ]:


# manifold_embedding(df_analysis, model, "log_l2")

# In[ ]:


# plot_manifold_neighbors("log_l2")

# In[ ]:


# surrogate_results = surrogate_rationalization(df_analysis, model, "log_l2")

# In[ ]:


# ranking_disagreement("log_l2")

# In[ ]:


# plot_ranking_disagreement_heatmap("log_l2")

# In[ ]:


# plot_importance_scatter("log_l2")

# In[ ]:


# attribution_inconsistency(df_analysis, model, "log_l2")

# In[ ]:


# manifold_embedding(df_analysis, model, "log_l2")

# In[ ]:


# shap_importance_stability(df_analysis, model, "log_l1")

# In[ ]:


# gaussian_noise_stability(df_analysis, model, "log_l2")

# In[ ]:


# pred_log_l2 = generate_predictions_from_pkl("log_l2", X_test, y_test)
# pred_log_l1 = generate_predictions_from_pkl("log_l1", X_test, y_test)

# In[ ]:


# # plot_model_diagnostics(y_test, y_prob, y_pred, "log_l2")
# plot_model_diagnostics(y_test, y_prob, y_pred, "log_l1")

# In[ ]:


# import joblib

# bundle = joblib.load("models/log_l1.pkl")

# model = bundle["model"]
# feature_cols = bundle["feature_cols"]

# df_analysis = X_test[feature_cols].copy()
# df_analysis["Label"] = y_test

# In[ ]:


# perm_df, shap_imp = analyze_baseline(df_analysis, model, "log_l2")

# In[ ]:


# generate_shap_waterfall(
#     model,
#     X_test,
#     y_test,
#     y_pred,
#     "log_l2"
# )

# In[ ]:


# attribution_inconsistency(df_analysis, model)

# In[ ]:


# manifold_embedding(df_analysis, model)
