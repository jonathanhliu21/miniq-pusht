from ml_collections import config_dict
import yaml
from pathlib import Path


def load_config(path: Path) -> config_dict.ConfigDict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config = config_dict.ConfigDict(cfg)

    extra_str = config.get("env", {}).get("extra_configs", "")
    if extra_str:
        extra_path = Path(extra_str)
        if not extra_path.is_absolute():
            extra_path = path.parent / extra_path
        with extra_path.open("r", encoding="utf-8") as f:
            extra_data = yaml.safe_load(f)
        if extra_data:
            config.update(extra_data)

    return config
