import os
import joblib

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

from work.paths import *
from work.custom_logger import get_logger

logger = get_logger()




# ============================================================
# MODEL CROSS-VALIDATION
# ------------------------------------------------------------
# Performs stratified k-fold cross-validation for a model and
# evaluates performance using ROC-AUC.
#
# The function:
# - splits the training data into stratified folds
# - trains the model on each fold
# - evaluates predictions on the validation fold
# - computes mean and standard deviation of AUC
#
# Output:
# cv_auc_mean  -> average AUC across folds
# cv_auc_std   -> standard deviation of AUC
#
# Interpretation:
# Cross-validation estimates how well the model generalizes
# to unseen data. Lower variance between folds suggests more
# stable model performance.
# ============================================================

def cross_validate_model(model, X, y, folds=3, seed=42):

    logger.info("Starting cross validation")

    # ============================================
    # STRATIFIED K-FOLD SPLIT
    # ============================================

    skf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed
    )

    X_np = X.to_numpy(copy=False)
    y_np = y.to_numpy(copy=False)

    auc_scores = []

    # ============================================
    # CROSS-VALIDATION LOOP
    # ============================================

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_np, y_np),
        start=1
    ):

        logger.info(f"CV fold {fold}/{folds}")

        X_tr = X_np[train_idx]
        y_tr = y_np[train_idx]

        X_val = X_np[val_idx]
        y_val = y_np[val_idx]

        # clone model to avoid leakage between folds
        m = clone(model)

        logger.info("Starting CV model fit")

        m.fit(X_tr, y_tr)

        logger.info("CV model fit finished")

        # ========================================
        # VALIDATION PERFORMANCE
        # ========================================

        y_pred = m.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_pred)

        logger.info(f"Fold {fold} AUC: {auc:.4f}")

        auc_scores.append(auc)

    # ============================================
    # FINAL CROSS-VALIDATION METRICS
    # ============================================

    cv_mean = np.mean(auc_scores)
    cv_std = np.std(auc_scores)

    logger.info(
        f"CV mean AUC: {cv_mean:.4f} (+/- {cv_std:.4f})"
    )

    return cv_mean, cv_std



# ============================================================
# MODEL EVALUATION ON TEST SET
# ------------------------------------------------------------
# Evaluates a trained model on the test dataset.
#
# The function:
# - generates predicted probabilities
# - converts probabilities to class predictions
# - computes standard classification metrics
#
# Metrics computed:
# AUC, accuracy, precision, recall, F1 score
#
# Output:
# results  -> dictionary with evaluation metrics
# y_prob   -> predicted probabilities
# y_pred   -> predicted class labels
# cm       -> confusion matrix
#
# Interpretation:
# These metrics describe final model performance on unseen
# data. AUC measures ranking ability, while the other metrics
# evaluate classification quality at the decision threshold.
# ============================================================

def evaluate_model(model, X_test, y_test):

    logger.info("Evaluating model on test set")

    # ============================================
    # GENERATE PREDICTIONS
    # ============================================

    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    # ============================================
    # COMPUTE PERFORMANCE METRICS
    # ============================================

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

    # ============================================
    # CONFUSION MATRIX
    # ============================================

    cm = confusion_matrix(y_test, y_pred)

    logger.info("Finished evaluating model")

    return results, y_prob, y_pred, cm






# ============================================================
# SAVE TRAINED MODEL AND METRICS
# ------------------------------------------------------------
# Saves the trained model together with metadata needed for
# later analysis.
#
# The saved bundle contains:
# - trained model object
# - list of feature columns used during training
# - model evaluation metrics
#
# Output files:
# models/MODEL_NAME.pkl
# results/metrics.csv
#
# Interpretation:
# The model bundle allows the analysis pipeline to later load
# the exact model and feature ordering used during training.
# The metrics file stores a summary of model performance so
# different models can be compared.
# ============================================================

def save_model_bundle(model, feature_cols, model_name, metrics):

    # ensure output directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ============================================
    # CREATE MODEL BUNDLE
    # ============================================

    bundle = {
        "model": model,
        "feature_cols": list(feature_cols),
        "metrics": metrics
    }

    # ============================================
    # SAVE MODEL FILE
    # ============================================

    model_path = os.path.join(
        MODELS_DIR,
        f"{model_name}.pkl"
    )

    joblib.dump(bundle, model_path)

    logger.info(f"Model saved: {model_path}")

    # ============================================
    # SAVE METRICS TABLE
    # ============================================

    metrics_df = pd.DataFrame([{
        "model": model_name,
        **metrics
    }])

    metrics_path = os.path.join(
        RESULTS_DIR,
        "metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        mode="a",
        header=not os.path.exists(metrics_path),
        index=False
    )

    logger.info(f"Metrics saved: {metrics_path}")




# ============================================================
# TRAIN LOGISTIC REGRESSION (L2 REGULARIZATION)
# ------------------------------------------------------------
# Trains a logistic regression model with L2 penalty.
#
# Steps:
# 1. perform cross-validation to estimate generalization
# 2. fit the final model on the full training set
# 3. evaluate the model on the test set
# 4. save the trained model and metrics
#
# Output files:
# models/log_l2.pkl
# results/metrics.csv
#
# Interpretation:
# Logistic regression is a linear baseline model. L2 penalty
# shrinks coefficients and reduces overfitting. This model
# serves as an interpretable reference for comparison with
# more complex models.
# ============================================================

def train_logistic_l2(X_train, y_train, X_test, y_test):

    logger.info("Training model: log_l2")

    # ============================================
    # DEFINE MODEL
    # ============================================

    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        tol=1e-4,
        random_state=42
    )

    # ============================================
    # CROSS-VALIDATION
    # ============================================

    cv_mean, cv_std = cross_validate_model(
        model,
        X_train,
        y_train
    )

    # ============================================
    # FINAL MODEL TRAINING
    # ============================================

    logger.info("Starting final model fit")

    model.fit(X_train, y_train)

    logger.info("Final model fit finished")

    # ============================================
    # TEST SET EVALUATION
    # ============================================

    test_metrics, y_prob, y_pred, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    # ============================================
    # COMBINE METRICS
    # ============================================

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    # ============================================
    # SAVE MODEL AND RESULTS
    # ============================================

    save_model_bundle(
        model,
        X_train.columns,
        "log_l2",
        results
    )

    logger.info("Model log_l2 training finished")

    return model



# ============================================================
# TRAIN LOGISTIC REGRESSION (L1 REGULARIZATION)
# ------------------------------------------------------------
# Trains a logistic regression model with L1 penalty.
#
# Steps:
# 1. perform cross-validation to estimate generalization
# 2. fit the final model on the full training set
# 3. evaluate the model on the test set
# 4. save the trained model and metrics
#
# Output files:
# models/log_l1.pkl
# results/metrics.csv
#
# Interpretation:
# L1 regularization encourages sparse solutions by pushing
# some coefficients to zero. This can act as implicit feature
# selection and may produce a simpler and more interpretable
# model compared to L2 logistic regression.
# ============================================================

def train_logistic_l1(X_train, y_train, X_test, y_test):

    logger.info("Training model: log_l1")

    # ============================================
    # DEFINE MODEL
    # ============================================

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.2,
        max_iter=1000,
        tol=1e-4,
        random_state=42
    )

    # ============================================
    # CROSS-VALIDATION
    # ============================================

    cv_mean, cv_std = cross_validate_model(
        model,
        X_train,
        y_train
    )

    # ============================================
    # FINAL MODEL TRAINING
    # ============================================

    logger.info("Starting final model fit")

    model.fit(X_train, y_train)

    logger.info("Final model fit finished")

    # ============================================
    # TEST SET EVALUATION
    # ============================================

    test_metrics, y_prob, y_pred, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    # ============================================
    # COMBINE METRICS
    # ============================================

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    # ============================================
    # SAVE MODEL AND RESULTS
    # ============================================

    save_model_bundle(
        model,
        X_train.columns,
        "log_l1",
        results
    )

    logger.info("Model log_l1 training finished")

    return model



# ============================================================
# TRAIN LOGISTIC REGRESSION (ELASTIC NET REGULARIZATION)
# ------------------------------------------------------------
# Trains a logistic regression model with elastic-net penalty,
# which combines L1 and L2 regularization.
#
# Because the EMBER dataset is large, training is performed on
# sampled subsets to reduce computation time.
#
# Steps:
# 1. sample subset of training data for cross-validation
# 2. sample larger subset for final model training
# 3. train the model
# 4. evaluate performance on the full test set
# 5. save the trained model and metrics
#
# Output files:
# models/elastic_net.pkl
# results/metrics.csv
#
# Interpretation:
# Elastic Net combines sparsity from L1 with stability from L2.
# Sampling allows faster training while still approximating
# model performance on large datasets.
# ============================================================

def train_elastic_net(X_train, y_train, X_test, y_test):

    logger.info("Training model: elastic_net")

    # ============================================
    # DEFINE MODEL
    # ============================================

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.3,
        C=0.5,
        max_iter=200,
        tol=1e-3,
        random_state=42
    )

    # ============================================
    # SAMPLE DATA FOR CROSS-VALIDATION
    # ============================================

    cv_size = min(50000, len(X_train))

    X_cv = X_train.sample(
        n=cv_size,
        random_state=42
    )

    y_cv = y_train.loc[X_cv.index]

    logger.info(
        f"Elastic net CV sample size: {X_cv.shape}"
    )

    cv_mean, cv_std = cross_validate_model(
        model,
        X_cv,
        y_cv
    )

    # ============================================
    # SAMPLE DATA FOR FINAL TRAINING
    # ============================================

    final_size = min(200000, len(X_train))

    X_final = X_train.sample(
        n=final_size,
        random_state=42
    )

    y_final = y_train.loc[X_final.index]

    logger.info(
        f"Elastic net final training sample size: {X_final.shape}"
    )

    # ============================================
    # FINAL MODEL TRAINING
    # ============================================

    logger.info("Starting final model fit")

    model.fit(X_final, y_final)

    logger.info("Final model fit finished")

    # report number of iterations used
    if hasattr(model, "n_iter_"):
        logger.info(
            f"Elastic net iterations: {model.n_iter_[0]}"
        )

    # ============================================
    # TEST SET EVALUATION
    # ============================================

    test_metrics, y_prob, y_pred, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    # ============================================
    # COMBINE METRICS
    # ============================================

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    # ============================================
    # SAVE MODEL AND RESULTS
    # ============================================

    save_model_bundle(
        model,
        X_train.columns,
        "elastic_net",
        results
    )

    logger.info("Model elastic_net training finished")

    return model




# ============================================================
# TRAIN RANDOM FOREST CLASSIFIER
# ------------------------------------------------------------
# Trains a Random Forest model on the EMBER dataset.
#
# Steps:
# 1. perform cross-validation on the training set
# 2. train the final model on the full training data
# 3. evaluate performance on the test set
# 4. save the trained model and metrics
#
# Output files:
# models/random_forest.pkl
# results/metrics.csv
#
# Interpretation:
# Random Forest is an ensemble of decision trees trained on
# bootstrapped samples of the data. The model captures
# nonlinear relationships and interactions between features.
# It also provides feature importance values that can be used
# in later explanation analyses.
# ============================================================

def train_random_forest(X_train, y_train, X_test, y_test):

    logger.info("Training model: random_forest")

    # ============================================
    # DEFINE MODEL
    # ============================================

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

    # ============================================
    # CROSS-VALIDATION
    # ============================================

    cv_mean, cv_std = cross_validate_model(
        model,
        X_train,
        y_train
    )

    # ============================================
    # FINAL MODEL TRAINING
    # ============================================

    logger.info("Starting final model fit")

    model.fit(X_train, y_train)

    logger.info("Final model fit finished")

    # ============================================
    # TEST SET EVALUATION
    # ============================================

    test_metrics, y_prob, y_pred, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    # ============================================
    # COMBINE METRICS
    # ============================================

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    # ============================================
    # SAVE MODEL AND RESULTS
    # ============================================

    save_model_bundle(
        model,
        X_train.columns,
        "random_forest",
        results
    )

    logger.info("Model random_forest training finished")

    return model



# ============================================================
# TRAIN LIGHTGBM GRADIENT BOOSTING MODEL
# ------------------------------------------------------------
# Trains a LightGBM gradient boosting classifier.
#
# Steps:
# 1. perform cross-validation on the training data
# 2. train the final model with early stopping
# 3. evaluate the model on the test set
# 4. save the trained model and metrics
#
# Output files:
# models/lightgbm.pkl
# results/metrics.csv
#
# Interpretation:
# LightGBM is a fast gradient boosting model that builds
# trees sequentially to reduce prediction error. Early
# stopping prevents overfitting by stopping training when
# validation performance stops improving.
# ============================================================

def train_lightgbm(X_train, y_train, X_test, y_test):

    logger.info("Training model: lightgbm")

    # ============================================
    # DEFINE MODEL
    # ============================================

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

    # ============================================
    # CROSS-VALIDATION
    # ============================================

    cv_mean, cv_std = cross_validate_model(
        model,
        X_train,
        y_train
    )

    # ============================================
    # FINAL MODEL TRAINING
    # ============================================

    logger.info("Starting final model fit")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )

    logger.info("Final model fit finished")

    # ============================================
    # TEST SET EVALUATION
    # ============================================

    test_metrics, y_prob, y_pred, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    # ============================================
    # COMBINE METRICS
    # ============================================

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    # ============================================
    # SAVE MODEL AND RESULTS
    # ============================================

    save_model_bundle(
        model,
        X_train.columns,
        "lightgbm",
        results
    )

    logger.info("Model lightgbm training finished")

    return model



# ============================================================
# TRAIN XGBOOST GRADIENT BOOSTING MODEL
# ------------------------------------------------------------
# Trains an XGBoost classifier on the EMBER dataset.
#
# Steps:
# 1. perform cross-validation on the training set
# 2. train the final model on the full training data
# 3. evaluate performance on the test set
# 4. save the trained model and metrics
#
# Output files:
# models/xgboost.pkl
# results/metrics.csv
#
# Interpretation:
# XGBoost is a gradient boosting model that builds decision
# trees sequentially to minimize prediction error. It is
# designed for high predictive performance on structured
# tabular data and often performs strongly on large datasets.
# ============================================================

def train_xgboost(X_train, y_train, X_test, y_test):

    logger.info("Training model: xgboost")

    # ============================================
    # DEFINE MODEL
    # ============================================

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

    # ============================================
    # CROSS-VALIDATION
    # ============================================

    logger.info("Starting cross validation")

    cv_mean, cv_std = cross_validate_model(
        model,
        X_train,
        y_train
    )

    # ============================================
    # FINAL MODEL TRAINING
    # ============================================

    logger.info("Starting final model fit")

    model.fit(
        X_train,
        y_train
    )

    logger.info("Final model fit finished")

    # ============================================
    # TEST SET EVALUATION
    # ============================================

    test_metrics, y_prob, y_pred, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    # ============================================
    # COMBINE METRICS
    # ============================================

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    # ============================================
    # SAVE MODEL AND RESULTS
    # ============================================

    save_model_bundle(
        model,
        X_train.columns,
        "xgboost",
        results
    )

    logger.info("Model xgboost training finished")

    return model



# ============================================================
# TRAIN MULTI-LAYER PERCEPTRON (NEURAL NETWORK)
# ------------------------------------------------------------
# Trains a feed-forward neural network classifier.
#
# Steps:
# 1. perform cross-validation on the training data
# 2. train the final neural network on the full training set
# 3. evaluate performance on the test set
# 4. save the trained model and metrics
#
# Output files:
# models/mlp.pkl
# results/metrics.csv
#
# Interpretation:
# The MLP model is a neural network with two hidden layers.
# It can capture complex nonlinear relationships between
# features that linear models cannot represent.
# Early stopping is used to reduce overfitting during training.
# ============================================================

def train_mlp(X_train, y_train, X_test, y_test):

    logger.info("Training model: mlp")

    # ============================================
    # DEFINE MODEL
    # ============================================

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

    # ============================================
    # CROSS-VALIDATION
    # ============================================

    cv_mean, cv_std = cross_validate_model(
        model,
        X_train,
        y_train
    )

    # ============================================
    # FINAL MODEL TRAINING
    # ============================================

    logger.info("Starting final model fit")

    model.fit(X_train, y_train)

    logger.info("Final model fit finished")

    # ============================================
    # TEST SET EVALUATION
    # ============================================

    test_metrics, y_prob, y_pred, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    # ============================================
    # COMBINE METRICS
    # ============================================

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    # ============================================
    # SAVE MODEL AND RESULTS
    # ============================================

    save_model_bundle(
        model,
        X_train.columns,
        "mlp",
        results
    )

    logger.info("Model mlp training finished")

    return model




# ============================================================
# TRAIN LINEAR SUPPORT VECTOR MACHINE
# ------------------------------------------------------------
# Trains a linear SVM classifier. Because LinearSVC does not
# provide probability estimates, it is wrapped with
# CalibratedClassifierCV to produce calibrated probabilities.
#
# Steps:
# 1. perform cross-validation on the training set
# 2. train the final calibrated SVM model
# 3. evaluate performance on the test set
# 4. save the trained model and metrics
#
# Output files:
# models/linear_svm.pkl
# results/metrics.csv
#
# Interpretation:
# Linear SVM is a linear classifier that finds a decision
# boundary maximizing the margin between classes. Probability
# calibration allows the model to produce probability scores,
# which are required for AUC evaluation and later XAI methods.
# ============================================================

def train_svm(X_train, y_train, X_test, y_test):

    logger.info("Training model: linear_svm")

    # ============================================
    # DEFINE BASE SVM MODEL
    # ============================================

    base = LinearSVC(
        C=1.0,
        max_iter=5000,
        tol=1e-4,
        random_state=42
    )

    # ============================================
    # CALIBRATE PROBABILITIES
    # ============================================

    model = CalibratedClassifierCV(
        base,
        method="sigmoid",
        cv=3
    )

    # ============================================
    # CROSS-VALIDATION
    # ============================================

    cv_mean, cv_std = cross_validate_model(
        model,
        X_train,
        y_train
    )

    # ============================================
    # FINAL MODEL TRAINING
    # ============================================

    logger.info("Starting final model fit")

    model.fit(X_train, y_train)

    logger.info("Final model fit finished")

    # ============================================
    # TEST SET EVALUATION
    # ============================================

    test_metrics, y_prob, y_pred, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    # ============================================
    # COMBINE METRICS
    # ============================================

    results = {
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        **test_metrics
    }

    # ============================================
    # SAVE MODEL AND RESULTS
    # ============================================

    save_model_bundle(
        model,
        X_train.columns,
        "linear_svm",
        results
    )

    logger.info("Model linear_svm training finished")

    return model




# ============================================================
# TRAIN ALL MODELS
# ------------------------------------------------------------
# Runs training for all models used in the experiment.
#
# The function sequentially trains:
# - logistic regression (L2)
# - logistic regression (L1)
# - elastic net logistic regression
# - random forest
# - LightGBM
# - XGBoost
# - multilayer perceptron (MLP)
# - linear SVM
#
# Each training function:
# - performs cross-validation
# - trains the final model
# - evaluates on the test set
# - saves the trained model and metrics
#
# Output:
# models dictionary containing trained model objects
#
# Interpretation:
# This function serves as the main entry point for model
# training in the experiment. It ensures that all models are
# trained using the same datasets and evaluation pipeline,
# allowing fair comparison of performance and interpretability.
# ============================================================

def train_all_models(X_train, y_train, X_test, y_test):

    logger.info("Starting training of all models")

    models = {}

    models["log_l2"] = train_logistic_l2(
        X_train, y_train, X_test, y_test
    )

    models["log_l1"] = train_logistic_l1(
        X_train, y_train, X_test, y_test
    )

    models["elastic_net"] = train_elastic_net(
        X_train, y_train, X_test, y_test
    )

    models["random_forest"] = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    models["lightgbm"] = train_lightgbm(
        X_train, y_train, X_test, y_test
    )

    models["xgboost"] = train_xgboost(
        X_train, y_train, X_test, y_test
    )

    models["mlp"] = train_mlp(
        X_train, y_train, X_test, y_test
    )

    models["linear_svm"] = train_svm(
        X_train, y_train, X_test, y_test
    )

    logger.info("All models trained")

    return models