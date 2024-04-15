from ai_service_interface import AIServiceInterface
from openai_adapter import OpenAIAdapter
from gemini_adapter import GeminiAdapter
from huggingface_adapter import HuggingFaceAdapter
from mistral_adapter import MistralAdapter

class AIServiceFactory:
    _service = None

    @staticmethod
    def set_ai_service(service):
        """
        Sets the AI service to the provided service.
        Parameters:
            service: The AI service to be set.
        Returns:
            None
        """
        AIServiceFactory._service = service

    @staticmethod
    def get_ai_service() -> AIServiceInterface:
        if AIServiceFactory._service is None:
            try:
                config = ConfigEnvironmentModule()
                AIServiceFactory._service = config.create_ai_service()
            except ValueError as e:
                print(f"Error getting AI service: {e}")
                return None
        return AIServiceFactory._service
