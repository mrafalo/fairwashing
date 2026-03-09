# ============================================================
# EMBER DATA PREPARATION SCRIPT
# ------------------------------------------------------------
# This script prepares the EMBER dataset for model training.
#
# Steps performed:
# 1. load raw EMBER parquet files
# 2. remove unlabeled samples (Label = -1)
# 3. split features and labels
# 4. convert feature types to float32
# 5. save raw train/test datasets
# 6. fit a StandardScaler on training data
# 7. create scaled versions of the datasets
# 8. save scaled datasets and scaler
#
# Output files (saved in /data):
# X_train.parquet
# X_test.parquet
# y_train.parquet
# y_test.parquet
# X_train_scaled.parquet
# X_test_scaled.parquet
# scaler_standard.pkl
#
# Interpretation:
# Raw datasets are used for tree-based models
# (RandomForest, LightGBM, XGBoost).
#
# Scaled datasets are used for models that require
# normalized features (Logistic Regression, SVM, MLP).
#
# The saved scaler ensures consistent preprocessing
# between training and future experiments.
# ============================================================

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib


# ==========================================
# PATHS
# ==========================================

DATA_DIR = "data"

TRAIN_FILE = os.path.join(DATA_DIR, "train_ember_2018_v2_features.parquet")
TEST_FILE  = os.path.join(DATA_DIR, "test_ember_2018_v2_features.parquet")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.parquet")
X_TEST_PATH  = os.path.join(DATA_DIR, "X_test.parquet")

Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.parquet")
Y_TEST_PATH  = os.path.join(DATA_DIR, "y_test.parquet")

X_TRAIN_SCALED_PATH = os.path.join(DATA_DIR, "X_train_scaled.parquet")
X_TEST_SCALED_PATH  = os.path.join(DATA_DIR, "X_test_scaled.parquet")

SCALER_PATH = os.path.join(DATA_DIR, "scaler_standard.pkl")


# ==========================================
# LOAD DATA
# ==========================================

print("Loading EMBER datasets...")

train = pd.read_parquet(TRAIN_FILE)
test  = pd.read_parquet(TEST_FILE)

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# ==========================================
# REMOVE UNLABELED SAMPLES
# ==========================================

train = train[train["Label"] != -1]
test  = test[test["Label"] != -1]

print("After removing unlabeled:")
print("Train:", train.shape)
print("Test:", test.shape)


# ==========================================
# SPLIT FEATURES AND LABEL
# ==========================================

y_train = train["Label"]
X_train = train.drop(columns=["Label"])

y_test = test["Label"]
X_test = test.drop(columns=["Label"])


# ==========================================
# CAST FEATURES TO FLOAT32
# ==========================================

X_train = X_train.astype("float32")
X_test  = X_test.astype("float32")


# ==========================================
# SAVE RAW DATASETS
# ==========================================

print("Saving raw datasets...")

X_train.to_parquet(X_TRAIN_PATH)
X_test.to_parquet(X_TEST_PATH)

y_train.to_frame("Label").to_parquet(Y_TRAIN_PATH)
y_test.to_frame("Label").to_parquet(Y_TEST_PATH)


# ==========================================
# STANDARD SCALING
# ==========================================

print("Fitting scaler...")

scaler = StandardScaler()
scaler.fit(X_train)

joblib.dump(scaler, SCALER_PATH)

print("Transforming datasets...")


X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# ==========================================
# SAVE SCALED DATASETS
# ==========================================

print("Saving scaled datasets...")

X_train_scaled.to_parquet(X_TRAIN_SCALED_PATH)
X_test_scaled.to_parquet(X_TEST_SCALED_PATH)

print("Done.")