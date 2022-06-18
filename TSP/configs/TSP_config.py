import yaml

def load_config(config_file):
    with open(config_file, "r") as fd:
        config = yaml.safe_load(fd)
    return config