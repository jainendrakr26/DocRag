"""Configuration module for RAG system."""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

class Config:
#API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#Model configurations
    LLM_MODEL = "openai:gpt-4o"

#Document processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

#Default URLS
    DEFAULT_URLS = [
        "https://raw.githubusercontent.com/jainendrakr26/python-numpy/main/dl.txt",
        "https://raw.githubusercontent.com/jainendrakr26/python-numpy/main/ml.txt",
        "https://medium.com/data-science/machine-learning-operations-mlops-for-beginners-a5686bfe02b2",
        #"https://medium.com/@RobuRishabh/introduction-to-machine-learning-555b0f1b62f5",
        #"https://medium.com/data-science/introducing-deep-learning-and-neural-networks-deep-learning-for-rookies-1-bd68f9cf5883"
]

    @classmethod
    def get_llm(cls):
        """
        Get the configured language model.

       """
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)
    