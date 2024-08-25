import json
import logging
from pathlib import Path
from typing import Any, Dict

import requests


def load_config(cfg_file: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        cfg_path = Path(cfg_file)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Configuration file '{cfg_file}' not found.")
        if not cfg_path.is_file():
            raise ValueError(f"'{cfg_file}' is not a valid file.")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return cfg
    except (yaml.YAMLError, FileNotFoundError) as e:
        logging.error(f"Error loading configuration file: {e}")
        raise


def load_db(db_file: str) -> Dict[str, Any]:
    """Load the project database from a JSON file."""
    try:
        db_path = Path(db_file)
        if not db_path.exists():
            return {}
        with open(db_path) as f:
            db = json.load(f)
        return db
    except json.JSONDecodeError as e:
        logging.error(f"Error loading database file: {e}")
        raise


def save_db(db: Dict[str, Any], db_file: str) -> None:
    """Save the project database to a JSON file."""
    try:
        db_path = Path(db_file)
        with open(db_path, "w") as f:
            json.dump(db, f, indent=4)
    except json.JSONEncodeError as e:
        logging.error(f"Error saving database file: {e}")
        raise


def stream_data(url: str) -> None:
    """Stream data from a given URL."""
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    yield chunk
    except requests.exceptions.RequestException as e:
        logging.error(f"Error streaming data from {url}: {e}")
        raise


def download_file(url: str, save_path: str) -> None:
    """Download a file from a given URL and save it to the specified path."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading file from {url}: {e}")
        raise