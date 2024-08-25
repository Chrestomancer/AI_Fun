import os
import logging
from typing import Dict, Any
from project_manager import ProjectManager
from utils import load_config, stream_data

logging.basicConfig(level=logging.INFO)

def main():
    """Ignite the Genesis Engine."""
    config = load_config("config.yaml")
    project_manager = ProjectManager(config)

    while True:
        idea = input("Enter your project idea (or 'quit' to exit): ")
        if idea.lower() == "quit":
            break

        try:
            project_manager.generate_project(idea)
            project_manager.visualize_project_graph()
        except Exception as e:
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()