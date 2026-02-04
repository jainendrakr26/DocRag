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

        # Check if any documents were retrieved
        if not state.retrieve_docs:
            return RAGState(
                questions=state.questions,
                retrieve_docs=state.retrieve_docs,
                answer="I don't have information about this in the documents."
            )

        context="\n".join([doc.page_content for doc in state.retrieve_docs])

        # Create prompt that enforces document-only answers
        prompt=f"""Answer the question ONLY based on the context provided below. 
If the answer cannot be found in the context, respond with exactly: "I don't have this information in the documents."

Do NOT use any general knowledge or information outside the provided context.

Context:
{context}

Question: {state.questions}"""
        
        # Generate response
        response=self.llm.invoke(prompt)

        return RAGState(
            questions=state.questions,
            retrieve_docs=state.retrieve_docs,
            answer=response.content
            )