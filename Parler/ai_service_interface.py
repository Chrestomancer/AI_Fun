import abc
import cachetools

class AIServiceInterface(abc.ABC):
    @abc.abstractmethod
    def send_request(self, input_data):
        pass

    @abc.abstractmethod
    async def send_request_async_impl(self, input_data):
        pass

    def send_request_with_cache(self, input_data):
        cache = cachetools.TTLCache(maxsize=128, ttl=300)
        cached_method = cachetools.cached(cache)(self.send_request)
        return cached_method(input_data)
