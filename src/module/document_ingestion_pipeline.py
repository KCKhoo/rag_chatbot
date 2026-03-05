from haystack import Document, Pipeline
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from constants import HF_EMBEDDING_MODEL, HF_TOKEN


class DocumentIngestionPipeline:
    def __init__(self, hf_embedding_model: str = HF_EMBEDDING_MODEL):
        """
        Initializes the document ingestion pipeline

        :param embedding_model: Embedding model on Hugging Face
        """

        self.embedder = HuggingFaceAPIDocumentEmbedder(
            api_type="serverless_inference_api",
            api_params={"model": hf_embedding_model},
            token=Secret.from_token(HF_TOKEN),
        )

        # Generate the embedding dimension, which is needed when initialising the
        # vector store in the next step
        self.embedding_dim = self._compute_embedding_dim()

        self.vector_store = QdrantDocumentStore(
            location=":memory:",
            similarity="cosine",
            embedding_dim=self.embedding_dim,
        )

        self.pipeline = self._create_pipeline()

    def _compute_embedding_dim(self) -> int:
        """Runs a single dummy document through the embedding model to find the embedding size."""

        result = self.embedder.run([Document(content="a")])
        return len(result["documents"][0].embedding)

    def _create_pipeline(self) -> Pipeline:
        """Create the document ingestion pipeline."""

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("indexing_embedder", self.embedder)
        indexing_pipeline.add_component(
            "indexing_writer",
            DocumentWriter(
                document_store=self.vector_store, policy=DuplicatePolicy.OVERWRITE
            ),
        )
        indexing_pipeline.connect("indexing_embedder", "indexing_writer")

        return indexing_pipeline

    def create_vector_store(self, documents: list[Document]) -> QdrantDocumentStore:
        """Ingests documents into the vector store."""

        self.pipeline.run({"indexing_embedder": {"documents": documents}})

        return self.vector_store
