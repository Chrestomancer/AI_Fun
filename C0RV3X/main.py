# main.py

import os
import logging
import sys
import argparse
from project import AdvancedProjectManager

# Configure logging
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, f'genius_{os.getpid()}.log')
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)  # Remove the existing log file (if any)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(description='Advanced Project Generation')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--idea', type=str, help='Project idea (required in non-interactive mode)')
    args = parser.parse_args()

    if not args.interactive and not args.idea:
        parser.error('--idea is required in non-interactive mode')

    project_manager = AdvancedProjectManager()

    if args.interactive:
        while True:
            idea = input("Project idea (or 'quit' to exit): ")
            if idea.lower() == 'quit':
                break
            logging.info(f"Generating project for idea: {idea}")
            project_manager.generate_project(idea)
    else:
        logging.info(f"Generating project for idea: {args.idea}")
        project_manager.generate_project(args.idea)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        sys.exit(1)