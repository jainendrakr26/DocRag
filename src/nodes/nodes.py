"""Langggraph nodes for RAG workflow"""

from src.state.rag_state import RAGState

class RAGNodes:
    """Containes Node functions for RAG workflow"""

    def __init__(self,retriever,llm):
        """Initialize RAG Nodes
        
        Args:
            retriever: The document retriever to use in the nodes.
            llm: The language model to use in the nodes."""
        self.retriever=retriever
        self.llm=llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Node function to retrieve documents based on the question in state.
        
        Args:
            state (RAGState): The current state containing the question.
        Returns: Updated RAGState with retrieved documents.
        """
        docs=self.retriever.invoke(state.questions)
        return RAGState(
            questions=state.questions,
            retrieve_docs=docs
            )
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """Node function to generate an answer based on retrieved documents.
        
        Args:
            state (RAGState): The current state containing the question and retrieved documents.
        Returns: Updated RAGState with generated answer.
        """

        context="\n".join([doc.page_content for doc in state.retrieve_docs])

        #create prompt
        prompt=f"""Answer the question based on the context below.
Context:
{context}
Question: {state.questions}"""
        
        #Generate response
        response=self.llm.invoke(prompt)

        return RAGState(
            questions=state.questions,
            retrieve_docs=state.retrieve_docs,
            answer=response.content
            )