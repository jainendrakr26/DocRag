"""Document processor module for handling document ingestion."""

from typing import List, Dict, Union
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
    PyPDFDirectoryLoader
    )

class DocumentProcessor:
    """Handles document loading and processing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize Document processor

        Args:
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlap size between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url: str) -> List[Document]:
        """
        Load document from a URL.

        Args:
            url (str): The URL of the document.
        """
        loader = WebBaseLoader(url)
        documents = loader.load()
        return documents
    
    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """
        Load document from all the PDF files in a directory.

        Args:
            file_path (str): The path to the PDF file.
        """
        loader = PyPDFDirectoryLoader(str(directory))
        documents = loader.load()
        return documents
    
    def load_from_text(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a text file.

        Args:
            file_path (str): The path to the text file.
        """
        loader = TextLoader(str(file_path), encoding='utf8')
        documents = loader.load()
        return documents
    
    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a PDF file.

        Args:
            file_path (str): The path to the PDF file.
        """
        loader = PyPDFDirectoryLoader(str("data"))
        return loader.load()
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from multiple sources.

        Args:
            sources (List[str]): List of file paths or URLs., PDF files, or text files.
        Returns:
            List[Document]: List of loaded documents.
        """
        docs:List[Document] = []
        for source in sources:
            if source.startswith("http://") or source.startswith("https://"):
                docs.extend(self.load_from_url(source))
            
            path=Path("data")
            if path.is_dir(): #PDF Directory
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_text(path))
            else:
                raise ValueError(f"Unsupported file type or source: {source}")
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents (List[Document]): List of documents to split.
        Returns:
            List[Document]: List of split document chunks.
        """
        return self.text_splitter.split_documents(documents)
    
    def process_url(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split document

        Args:
            urls (List[str]): list of urls to process.
        Returns:
            List[Document]: List of processed document chunks.
        """
        docs=self.load_documents(urls)
        return self.split_documents(docs)

                