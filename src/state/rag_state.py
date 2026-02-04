"""RAG state definition for LANGGRAPH"""

from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    """State model for Retrieval-Augmented Generation (RAG)"""

    questions: str
    retrieve_docs: List[Document] = []
    answer: str = ""