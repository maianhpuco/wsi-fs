import yaml 

def load_yaml_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config 