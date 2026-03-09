import pandas as pd
from sklearn.model_selection import train_test_split
import legacy.globals as g
import work.custom_logger as cl

logger = cl.get_logger()


def load_data():

    logger.info("Loading dataset")

    df = pd.read_csv(g.SAMPLE_FILE, sep=";")

    X = df.drop(["Label"], axis=1)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=g.SEED
    )

    logger.info(
        f"dataset size: {len(df)} | train: {len(X_train)} | test: {len(X_test)}"
    )

    return X_train, y_train, X_test, y_test