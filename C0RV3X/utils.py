# utils.py

import json
import logging
from pathlib import Path

def load_config(cfg_file):
    try:
        if not Path(cfg_file).exists():
            raise FileNotFoundError(f"Configuration file '{cfg_file}' not found.")
        if not Path(cfg_file).is_file():
            raise ValueError(f"'{cfg_file}' is not a valid file.")
        with open(cfg_file) as f:
            cfg = json.load(f)
        assert all(k in cfg['prompt_template'] for k in ["code", "refine", "relate_modules"]), "Missing prompt templates"
        return cfg
    except json.JSONDecodeError as e:
        logging.error(f"Configuration file '{cfg_file}' is not a valid JSON file: {e}")
    except AssertionError as e:
        logging.error(f"Configuration error: {e}")
        raise

def load_db(db_file):
    return json.load(open(db_file)) if Path(db_file).exists() else {}

def save_db(db, db_file):
    try:
        json.dump(db, open(db_file, 'w'))
    except FileNotFoundError as e:
        logging.error(f"Configuration file '{cfg_file}' not found: {e}")
        raise