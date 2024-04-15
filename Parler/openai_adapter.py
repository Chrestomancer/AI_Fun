import openai
from ai_service_interface import AIServiceInterface

class OpenAIAdapter(AIServiceInterface):
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.conversation_history = []

    def send_request(self, input_data):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "system", "content": "You are an engaging conversationalist that replies in 2-3 sentences."}] +
                         [{"role": "user", "content": i} for i in self.conversation_history[-5:]] +
                         [{"role": "user", "content": input_data}],
                temperature=0.91,
                max_tokens=333
            )
            self.conversation_history.append(response.choices[0].message.content)
            return response.choices[0].message.content.strip() if response.choices else None
        except Exception as e:
            print(f"Error in OpenAIAdapter: {e}")
            return None

    async def send_request_async_impl(self, input_data):
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "system", "content": "Plan and Generate Python code separating files and modules with format: # File: filename.py:"}] +
                         [{"role": "user", "content": i} for i in self.conversation_history[-5:]] +
                         [{"role": "user", "content": input_data}],
                temperature=0.61,
                max_tokens=1050
            )
            self.conversation_history.append(response.choices[0].message.content)
            return response.choices[0].message.content.strip() if response.choices else None
        except Exception as e:
            print(f"Error in OpenAIAdapter: {e}")
            return None
