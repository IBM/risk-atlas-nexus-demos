import json
from pathlib import Path
from typing import Dict


def resolve_file_paths(param_dict):
    for param, param_value in param_dict.items():
        if isinstance(param_value, str) and param_value.lower().endswith(".json"):
            param_dict[param] = json.load(Path(param_value).open("r"))
        elif isinstance(param_value, Dict):
            resolve_file_paths(param_value)
