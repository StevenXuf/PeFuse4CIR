import yaml

def get_default_config(config_path='./config.yaml'):
    with open(config_path,'r') as f:
        return yaml.safe_load(f)
