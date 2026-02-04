"""Graph builder for langgraph workflow"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes

class GraphBuilder:
    """Graph builderbuilds and manages a langgraph workflow"""

    def __init__(self,retriever,llm):
        """Initialize Graph Builder
        
        Args:
            retriever: The document retriever to use in the graph.
            llm: The language model to use in the graph."""
        self.nodes=RAGNodes(retriever,llm)
        self.graph=None

    def build(self) -> StateGraph:
        """Builds a graph for Retrieval-Augmented Generation (RAG) workflow
        Returns:
            StateGraph: The constructed state graph for RAG.
        """

        #create a state graph
        builder=StateGraph(RAGState)
        
        #Add Nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)

        #Set entry point
        builder.set_entry_point("retriever")

        #Add edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

       #Compile graph
        self.graph=builder.compile()
        return self.graph
    
    def run(self, question: str) -> dict:
        """Run the RAG flow
        
        Args:
            question (str): User question.
        Returns:
            RAGState: Final state with answer.
        """
        if self.graph is None:
            self.build()
        
        initial_state=RAGState(questions=question)
        return self.graph.invoke(initial_state)