from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from ai_service_interface import AIServiceInterface

class MistralAdapter(AIServiceInterface):
    def __init__(self, api_key, model_name="mistral-tiny"):
        self.client = MistralClient(api_key=api_key)
        self.model = model_name
        self.conversation_history = []

    def send_request(self, input_data, stream=False):
        self.conversation_history.append(ChatMessage(role="user", content=input_data))
        messages = [ChatMessage(role="system", content="Your system message here")] + self.conversation_history[-5:]

        try:
            if stream:
                for chunk in self.client.chat_stream(model=self.model, messages=messages):
                    print(chunk)
            else:
                chat_response = self.client.chat(model=self.model, messages=messages)
                if chat_response.choices:
                    last_choice = chat_response.choices[-1]
                    response_content = last_choice.message.content
                    self.conversation_history.append(ChatMessage(role="assistant", content=response_content))
                    return response_content.strip() if response_content else None
            return None
        except Exception as e:
            print(f"Error in MistralAdapter: {e}")
            return None

    async def send_request_async_impl(self, input_data):
        return self.send_request(input_data)
