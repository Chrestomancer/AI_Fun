# main.py

import os
import logging
import sys

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

from project import ProjectManager

def main():
    genius = ProjectManager()  # Initialize the project manager
    while True:
        idea = input("Project idea (or 'quit' to exit): ")
        if idea.lower() == 'quit':
            break
        genius.generate_project(idea)  # Generate the project based on the idea

if __name__ == '__main__':
    main()