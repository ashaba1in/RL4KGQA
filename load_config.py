import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

config = load_config('/content/drive/MyDrive/Diploma/configs/rl_config.yaml')
