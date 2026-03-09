import os

# ==============================
# BASE PATH
# ==============================

WORKSPACE_PATH = os.getcwd()

# ==============================
# MAIN DIRECTORIES
# ==============================

DATA_DIR = os.path.join(WORKSPACE_PATH, "data")
MODELS_DIR = os.path.join(WORKSPACE_PATH, "models")
RESULTS_DIR = os.path.join(WORKSPACE_PATH, "results")

# ==============================
# RESULTS SUBDIRECTORIES
# ==============================

PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
IMPORTANCE_DIR = os.path.join(RESULTS_DIR, "importance")
SURROGATE_DIR = os.path.join(RESULTS_DIR, "surrogate")
XAI_DIR = os.path.join(RESULTS_DIR, "xai")

# ==============================
# ENSURE DIRECTORIES EXIST
# ==============================

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(IMPORTANCE_DIR, exist_ok=True)
os.makedirs(SURROGATE_DIR, exist_ok=True)
os.makedirs(XAI_DIR, exist_ok=True)