from langchain_huggingface.llms import HuggingFacePipeline
from transformers import BitsAndBytesConfig
from llm.prompts import rag_prompt
from settings import get_settings

settings = get_settings()


nb_config = BitsAndBytesConfig(
    load_in_4bit=True,        
    bnb_4bit_compute_dtype="float16",  
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4"
)

class LLM:
    """Wrapper class for a HuggingFace LLM model using LangChain."""

    def __init__(self):
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=settings.llm,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 1024},
            model_kwargs={
                "quantization_config": nb_config},
        )
        self.prompt = rag_prompt

    def generate(self, query, context="") -> str:
        """Generate a response from the LLM given a query and optional context.

        Args:
            query (str): The input query string.
            context (str, optional): Additional context to provide to the LLM. Defaults to "".

        Returns:
            str: The generated response from the LLM.
        """
        message = self.prompt.format(question=query, context=context)
        response = self.llm.invoke(message)
        completion = response[len(message) :].strip()
        return completion
