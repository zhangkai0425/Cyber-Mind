import os

import yaml


def parse_config(config_path: str = None):
    if config_path is None:
        config_path = os.environ.get('LM_CONFIG', None)
    assert config_path is not None, "Config file path not found in argument or LM_CONFIG env var!"
    with open(config_path, 'r') as f:
        local_config = yaml.load(f, Loader=yaml.FullLoader)
    return local_config