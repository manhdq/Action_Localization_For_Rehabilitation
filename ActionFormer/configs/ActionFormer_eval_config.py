import yaml


DEFAULTS_CONFIG = 'ActionFormer/configs/thumos_default_AF_eval.yaml'


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    with open(DEFAULTS_CONFIG, 'r') as fd:
        try:
            config = yaml.load(fd, Loader=yaml.FullLoader)
        except:
            config = yaml.load(fd)
    return config

def _update_config(config):
    # fill in derived fields
    config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["num_classes"] = config["dataset"]["num_classes"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config

def load_config(config_file):
    defaults = load_default_config()
    with open(config_file, "r") as fd:
        try:
            config = yaml.load(fd, Loader=yaml.FullLoader)
        except:
            config = yaml.load(fd)
    _merge(defaults, config)
    config = _update_config(config)
    return config