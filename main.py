import pandas as pd
from pathlib import Path

from work.training import train_all_models, train_logistic_l2
from work.analysis import run_analysis_pipeline, build_model_summary
from work.custom_logger import get_logger


logger = get_logger()

print("MAIN FILE LOADED")

# =========================================================
# PROJECT PATHS
# =========================================================
# Determine project root and data directory automatically

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


# =========================================================
# FULL EXPERIMENT PIPELINE
# =========================================================
# Runs the complete experiment:
#
# 1. Load prepared EMBER datasets
# 2. Train all models
# 3. Run XAI analysis pipeline for every model
# 4. Build a comparison table summarizing model properties
#
# This is the version used for the final experiment.
# =========================================================

# def main():
#
#     logger.info("Starting fairwashing experiment")
#
#     # =================================
#     # LOAD DATA
#     # =================================
#
#     logger.info("Loading datasets")
#
#     X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
#     X_test  = pd.read_parquet(DATA_DIR / "X_test.parquet")
#
#     y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")["Label"]
#     y_test  = pd.read_parquet(DATA_DIR / "y_test.parquet")["Label"]
#
#     logger.info("Datasets loaded")
#
#     # =================================
#     # TRAIN MODELS
#     # =================================
#
#     models = train_all_models(
#         X_train,
#         y_train,
#         X_test,
#         y_test
#     )
#
#     logger.info("Model training finished")
#
#     # =================================
#     # RUN ANALYSIS FOR EACH MODEL
#     # =================================
#
#     for model_name in models.keys():
#
#         logger.info(f"Running analysis for {model_name}")
#
#         try:
#
#             run_analysis_pipeline(
#                 model_name,
#                 X_test,
#                 y_test,
#                 mode="fast"
#             )
#
#         except Exception as e:
#
#             logger.error(f"{model_name} failed: {e}")
#
#     # =================================
#     # BUILD MODEL COMPARISON TABLE
#     # =================================
#
#     build_model_summary()
#
#     logger.info("Experiment finished")
#
#
# if __name__ == "__main__":
#     main()


# =========================================================
# SMALL PIPELINE TEST
# =========================================================
# Quick pipeline test:
#
# - loads the prepared EMBER dataset
# - takes a very small random sample
# - trains a single model
# - runs the analysis pipeline
#
# This allows verifying that the entire pipeline works
# before launching the full experiment.
# =========================================================

def main():

    print("MAIN FUNCTION STARTED")

    logger.info("Starting PIPELINE TEST")

    # =================================
    # LOAD DATA
    # =================================

    logger.info("Loading datasets")

    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    X_test  = pd.read_parquet(DATA_DIR / "X_test.parquet")

    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")["Label"]
    y_test  = pd.read_parquet(DATA_DIR / "y_test.parquet")["Label"]

    logger.info("Datasets loaded")

    # =================================
    # CREATE SMALL SAMPLE
    # =================================

    X_train = X_train.sample(10000, random_state=42)
    y_train = y_train.loc[X_train.index]

    X_test = X_test.sample(5000, random_state=42)
    y_test = y_test.loc[X_test.index]

    logger.info("Using small test sample")

    # =================================
    # TRAIN SINGLE MODEL
    # =================================

    logger.info("Training small logistic model")

    train_logistic_l2(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # =================================
    # RUN ANALYSIS PIPELINE
    # =================================

    logger.info("Running pipeline analysis")

    run_analysis_pipeline(
        "log_l2",
        X_test,
        y_test,
        mode="fast"
    )

    logger.info("Pipeline test finished")


if __name__ == "__main__":
    main()