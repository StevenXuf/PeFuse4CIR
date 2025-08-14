import yaml
from omegaconf import OmegaConf
def get_default_config(config_path='./config.yaml'):
    with open(config_path,'r') as f:
        return yaml.safe_load(f)
    
def get_default_omegaconf(config_path='./config.yaml'):
    return OmegaConf.load(config_path)
