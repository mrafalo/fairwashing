# ============================================================
# LOGGER CONFIGURATION
# ------------------------------------------------------------
# Creates and configures a project-wide logger used to track
# progress of training and analysis steps.
#
# The logger outputs messages to:
# - console (terminal output)
# - log file: results/analysis.log
#
# Log messages include:
# timestamp | log level | message
#
# Interpretation:
# Logging allows monitoring experiment progress and helps
# diagnose errors during long-running model training and
# analysis pipelines.
# ============================================================

import logging
import os
from work.paths import RESULTS_DIR


def get_logger(name="analysis"):

    logger = logging.getLogger(name)

    # avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # console output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file output
    log_path = os.path.join(RESULTS_DIR, "analysis.log")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger