import os
import yaml

# ==============================
# FIND PROJECT ROOT
# ==============================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# ==============================
# LOAD CONFIG
# ==============================

with open(CONFIG_PATH, "r") as file:
    cfg = yaml.safe_load(file)

# ==============================
# GLOBAL VARIABLES
# ==============================

SEED = cfg["SEED"]
DATA_FILE = cfg["DATA_FILE"]
SAMPLE_FILE = cfg["SAMPLE_FILE"]