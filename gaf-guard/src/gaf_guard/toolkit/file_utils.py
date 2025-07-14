import json
from pathlib import Path
from typing import Dict


def resolve_file_paths(param_dict):
    for param, param_value in param_dict.items():
        if isinstance(param_value, str) and param_value.lower().endswith(".json"):
            param_dict[param] = json.load(Path(param_value).open("r"))
        elif isinstance(param_value, Dict):
            resolve_file_paths(param_value)


def extract_run_configs(param_dict):
    run_configs = {}
    for param_name, param_value in dict(param_dict).items():
        if isinstance(param_name, str) and param_name == "run_configs":
            run_configs = run_configs | param_dict.pop(param_name)
        elif isinstance(param_value, Dict):
            run_configs = run_configs | extract_run_configs(param_value)

    return run_configs
