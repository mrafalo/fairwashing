import argparse
import work
import work.models as m
import work.data as d
import work.globals as g
import work.reports as r
import importlib
import work.custom_logger as cl
import time

logger = cl.get_logger()


def model_iterator(_scenario):
    logger.info("model iterator...")
    df = d.load_data()
    X_train, y_train, X_test, y_test = d.get_data()
    res, m1 = m.model_random_forest(X_train, y_train, X_test, y_test)
    logger.info(f"model OK AUC: {res}")

    logger.info(f"surrogate rationalization")
    m.surrogate_rationalization(df, m1)

    logger.info(f"attribution inconsistency")
    m.attribution_inconsistency(df, m1)

    logger.info(f"mainfoild embedding")
    m.mainfold_embedding(df, m1)


def main_run(_mode, _scenario):
    start = time.time()
    logger.info(f"starting for mode {_mode}, scenario {_scenario}...")
    if _mode == "1":
        model_iterator(_scenario)
    stop = time.time()
    elapsed_sec = stop - start
    logger.info("finished!, elapsed: " + str(elapsed_sec // 60) + " minutes")


def main():
    parser = argparse.ArgumentParser(description="Mode and scenario")
    parser.add_argument("_mode", type=str, help="run mode")
    parser.add_argument("_scenario", type=str, help="scenario name")
    args = parser.parse_args()
    main_run(args._mode, args._scenario)


if __name__ == "__main__":
    main()

# importlib.reload(work.data)
# importlib.reload(work.models)
# importlib.reload(work.globals)
