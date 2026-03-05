from typing import Any

from module.response_generation_pipeline import ResponseGenerationPipeline
from module.retrieval_pipeline import RetrievalPipeline


class RAGPipeline:
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        response_generation_pipeline: ResponseGenerationPipeline,
    ):
        """
        Create an end-to-end RAG pipeline

        :param retrieval_pipeline: An instance of RetrievalPipeline
        :param generation_pipeline: An instance of ResponseGenerationPipeline
        """
        self.retrieval_pipeline = retrieval_pipeline
        self.response_generation_pipeline = response_generation_pipeline

    def answer(self, question: str) -> dict[str, Any]:
        """Answer user question"""

        try:
            retrieved_context = self.retrieval_pipeline.retrieve_contexts(question)
            answer = self.response_generation_pipeline.generate_response(
                question, retrieved_context
            )

            return {
                "status": "success",
                "error_message": None,
                "question": question,
                "answer": answer,
                "context": retrieved_context,
                # Future Improvement: Output latency, token usage for model monitoring
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "question": question,
                "answer": None,
                "context": None,
            }
