"""LangGraph nodes for RAG workflow"""

from src.state.rag_state import RAGState


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents from the vector store."""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer strictly from retrieved documents."""
        if not state.retrieved_docs:
            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer="No relevant documents were found for your question."
            )

        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        prompt = (
            "Answer the question using ONLY the context below. "
            "Do not use any outside knowledge. "
            "If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state.question}\n\n"
            "Answer:"
        )

        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )
