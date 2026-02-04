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
        "https://github.com/jainendrakr26/python-numpy/blob/main/ml.txt",
        "https://github.com/jainendrakr26/python-numpy/blob/main/dl.txt"
]

    @classmethod
    def get_llm(cls):
        """
        Get the configured language model.

       """
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)
    