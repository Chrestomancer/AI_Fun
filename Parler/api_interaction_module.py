from ai_service_factory import AIServiceFactory
from error_logging_module import ErrorLoggingModule
from config_environment_module import ConfigEnvironmentModule
from token_bucket import TokenBucket
class APIInteractionModule:
    def __init__(self, config_environment_module, error_logging_module):
        self.config = config_environment_module
        self.error_logging = error_logging_module
        self.ai_adapter = AIServiceFactory.get_ai_service()
        self.rate_limiter = TokenBucketImport(self.config.requests_per_second, self.config.burst_size)

    async def generate_content_async(self, prompt):
        try:
            print("Generating content based on provided prompt...")
            async with self.rate_limiter:
                response = await self.ai_adapter.send_request_async_impl(prompt)
            return response
        except Exception as e:
            self.error_logging.log_error(f"Error generating content: {e}")
            return "Sorry, I couldn't process your request."
