import yaml

with open(r"config.yaml") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    SEED = cfg["SEED"]
    DATA_FILE = cfg["DATA_FILE"]
    SAMPLE_FILE = cfg["SAMPLE_FILE"]
