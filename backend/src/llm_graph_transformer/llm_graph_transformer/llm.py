from .OpenAI_graph_transformer import *

class LLMGraphTransformer:
    def __init__(self,llm_model):
        self.llm_model=llm_model

    def get_llm_object(self,api_key):
        if 'gpt' in self.llm_model:
            return OpenaiGraphTransformer(openai_api_key=api_key)
        else:
            return 'Currently this llm model is not supported.'