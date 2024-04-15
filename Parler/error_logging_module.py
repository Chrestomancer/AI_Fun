import logging
from datetime import datetime

class ErrorLoggingModule:
    def __init__(self):
        self.setup_logging()

    @staticmethod
    def setup_logging():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        logging.basicConfig(level=logging.INFO,
                            filename=f'error_log_{timestamp}.log',
                            format='%(asctime)s | %(levelname)s | %(message)s')

    @staticmethod
    def log_error(message):
        logging.error(message)

    @staticmethod
    def log_warning(message):
        logging.warning(message)

    @staticmethod
    def log_info(message):
        logging.info(message)
