import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split
import work.custom_logger as cl
import glob
import work.globals as g
import pyarrow.parquet as pq
import polars as pl
import random

logger = cl.get_logger()


def load_data():
    df = pd.read_csv(g.SAMPLE_FILE, sep=";")
    return df


def get_data():
    df = load_data()

    X = df.drop(["Label"], axis=1)
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    res_X_train = X_train
    res_y_train = y_train
    res_X_test = X_test
    res_y_test = y_test

    logger.info(
        f"dataset size: {len(df)} | X train: {len(res_X_train)} | X test: {len(res_X_test)}"
    )

    return res_X_train, res_y_train, res_X_test, res_y_test


def prepare_data_file():
    df = pl.scan_parquet(g.DATA_FILE)

    # Get column names (requires schema)
    all_cols = df.collect_schema().names()

    # Remove Label from feature list
    feature_cols = [c for c in all_cols if c != "Label"]

    # Random 100 feature columns
    random_cols = random.sample(feature_cols, 20)

    # Final column selection (features + Label)
    selected_cols = random_cols + ["Label"]

    # Sample 100k rows
    sampled = df.select(selected_cols).limit(50_000).collect()
    # Save to CSV
    sampled.write_csv(g.SAMPLE_FILE, separator=";")
