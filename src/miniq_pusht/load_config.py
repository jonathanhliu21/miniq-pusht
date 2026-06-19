from ml_collections import config_dict
import yaml
from pathlib import Path

def load_config(path: Path) -> config_dict.ConfigDict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config = config_dict.ConfigDict(cfg)
    return config