from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
import os
from yacs.config import CfgNode as CN

from .default import _C

_VALID_TYPES = {tuple, list, str, int, float, bool}


def convert_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict

def load_config(config_file):
    default_dict = convert_to_dict(_C)
    with open(config_file, "r") as fd:
        config = yaml.safe_load(fd)
    # merge 2 dict
    config = default_dict | config
    # config = config | default_dict
    config['DATASET']['ROOT'] = os.path.join(config['DATA_DIR'], config['DATASET']['ROOT'])
    config['MODEL']['PRETRAINED'] = os.path.join(config['DATA_DIR'], config['MODEL']['PRETRAINED'])
    config['TEST']['MODEL_FILE'] = os.path.join(config['DATA_DIR'], config['TEST']['MODEL_FILE'])
    return config