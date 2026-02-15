from haystack import Pipeline
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.utils import Secret
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.constants import HF_TOKEN, HF_EMBEDDING_MODEL


class RetrievalPipeline:
    def __init__(
        self,
        document_store: QdrantDocumentStore,
        hf_embedding_model: str = HF_EMBEDDING_MODEL,
        top_k: int = 3,
    ):
        """
        Initializes the retrieval pipeline.

        :param document_store: QdrantDocumentStore containing FAQ embeddings.
        :param embedding_model: Embedding model on Hugging Face (must be the same embedding model used in document ingestion pipeline)
        :param top_k: Number of context to retrieve.
        """

        self.embedder = HuggingFaceAPITextEmbedder(
            api_type="serverless_inference_api",
            api_params={"model": hf_embedding_model},
            token=Secret.from_token(HF_TOKEN),
        )

        self.retriever = QdrantEmbeddingRetriever(
            document_store=document_store, top_k=top_k
        )

        self.pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        """Create the retrieval pipeline"""
        retrieval_pipeline = Pipeline()
        retrieval_pipeline.add_component("retrieval_embedder", self.embedder)
        retrieval_pipeline.add_component("retriever", self.retriever)
        retrieval_pipeline.connect(
            "retrieval_embedder.embedding", "retriever.query_embedding"
        )

        return retrieval_pipeline

    def retrieve_contexts(self, query: str):
        """Retrieve top K relevant context for a given query"""
        return self.pipeline.run({"retrieval_embedder": {"text": query}})["retriever"][
            "documents"
        ]
