from google.generativeai import GenerativeModel, configure as configure_genai
from ai_service_interface import AIServiceInterface

class GeminiAdapter(AIServiceInterface):
    def __init__(self, api_key):
        self.api_key = api_key
        configure_genai(api_key=self.api_key)
        self.model = GenerativeModel('gemini-pro')

    def send_request(self, input_data):
        try:
            response = self.model.generate_content(input_data)
            return response.text.strip() if response.text else None
        except Exception as e:
            print(f"Error in GeminiAdapter: {e}")
            return None

    async def send_request_async_impl(self, input_data):
        return self.send_request(input_data)
