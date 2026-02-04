"""Vector store module for managing document embeddings and retrieval."""

from typing import List, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    """Manages Vector Store operations."""
    
    def __init__(self):
        """
        Initialize Vector Store with an embedding model.

        Args:
            embedding_model (Any): The embedding model to use. Defaults to OpenAIEmbeddings.
        """
        self.embedding_model = OpenAIEmbeddings()
        self.vector_store = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create a vector store from the provided documents.

        Args:
            documents (List[Document]): The list of documents to index.
        """
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        # Use similarity_search_with_score to get relevance scores
        # This enables filtering by relevance threshold
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 similar documents
        )

    def get_retriever(self):
        """
        Get the retriever for querying the vector store.

        Returns:
            Any: The retriever object.
        """
        if self.retriever is None:
            raise ValueError("Retriever has not been created. Call create_retriever first.")
        return self.retriever
    
    def retrieve(self, query: str, k:int=4) -> List[Document]:
        """
        Retrieve documents for the query.

        Args:
            query (str): The query string.
            k (int): Number of documents to retrieve. Defaults to 4.

        Returns:
            List[Document]: The list of retrieved documents.
        """
        if self.retriever is None:
            raise ValueError("Retriever has not been created. Call create_retriever first.")
        return self.retriever.invoke(query)