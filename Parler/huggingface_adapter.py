from ai_service_interface import AIServiceInterface
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class HuggingFaceAdapter(AIServiceInterface):
    def __init__(self, model_name="microsoft/phi-2", max_length=50, use_pipeline=False):
        self.model_name = model_name
        self.max_length = max_length
        self.use_pipeline = use_pipeline
        self.model = None
        self.tokenizer = None
        self.generator_pipeline = None
        self.device = self.detect_gpu()

    def detect_gpu(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self):
        if self.use_pipeline:
            self.load_pipeline()
        else:
            self.load_model_and_tokenizer()

    def load_pipeline(self):
        if self.generator_pipeline is None:
            self.generator_pipeline = pipeline("text-generation", model=self.model_name, device=0 if self.device != "cpu" else -1)

    def load_model_and_tokenizer(self):
        if self.model is None or self.tokenizer is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def send_request(self, input_data):
        self.load_model()
        if self.use_pipeline and self.generator_pipeline is None:
            return "Pipeline loading failed."
        if not self.use_pipeline and (self.model is None or self.tokenizer is None):
            return "Model and tokenizer loading failed."

        try:
            if self.use_pipeline:
                outputs = self.generator_pipeline(input_data, max_length=self.max_length)
                return outputs[0]["generated_text"]
            else:
                inputs = self.tokenizer(input_data, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, max_length=self.max_length)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    async def send_request_async_impl(self, input_data):
        return self.send_request(input_data)
