from ai_service_interface import AIServiceInterface
from openai_adapter import OpenAIAdapter
from gemini_adapter import GeminiAdapter
from huggingface_adapter import HuggingFaceAdapter
from mistral_adapter import MistralAdapter

class AIServiceFactory:
    _service = None

    @staticmethod
    def set_ai_service(service):
        AIServiceFactory._service = service

    @staticmethod
    def get_ai_service() -> AIServiceInterface:
        if AIServiceFactory._service is None:
            raise ValueError("AI service not set")
        return AIServiceFactory._service
