import os
import logging
import platform
from ai_service_factory import AIServiceFactory
from openai_adapter import OpenAIAdapter


class ConfigEnvironmentModule:
    def __init__(self):
        self.environment = self.detect_environment()
        logging.info(f"Detected environment: {self.environment}")
        self.os_type = self.detect_os_and_system_info()
        logging.info(f"Operating System: {self.os_type}")

        self.api_key = self.load_api_key()
        self.ai_service = self.detect_ai_service()

        AIServiceFactory.set_ai_service(self.create_ai_service())

    @staticmethod
    def detect_environment():
        return os.getenv('ENVIRONMENT', 'development').lower()

    @staticmethod
    def detect_os_and_system_info():
        return platform.system()

    def detect_ai_service(self):
        service_priority = ['openai', 'gemini', 'mistral', 'huggingface']
        for service in service_priority:
            if self.is_service_available(service):
                return service
        raise ValueError("No AI services available.")

    def is_service_available(self, service_name):
        api_key = self.load_api_key(service_name)
        return api_key is not None

    def load_api_key(self, service_name=None):
        if service_name:
            env_var_name = f'{service_name.upper()}_API_KEY'
        else:
            env_var_name = 'AI_API_KEY'

        api_key = os.getenv(env_var_name)
        if api_key:
            return api_key

        if service_name:
            default_path = "api_keys"
            file_path = os.path.join(default_path, f'{service_name}_api_key.txt')
            return self.load_api_key_from_file(file_path)
        return None

    @staticmethod
    def load_api_key_from_file(file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read().strip()
        except IOError:
            logging.error(f"Unable to read API key from file: {file_path}")
            return None

    def create_ai_service(self):
        service_name = self.ai_service
        if service_name == "openai":
            return OpenAIAdapter(self.api_key)
        elif service_name == "gemini":
            return GeminiAdapter(self.api_key)
        elif service_name == "mistral":
            return MistralAdapter(self.api_key)
        elif service_name == "huggingface":
            return HuggingFaceAdapter()
        else:
            raise ValueError(f"Unsupported AI service: {service_name}")
