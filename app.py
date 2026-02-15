import os
import streamlit as st
from haystack import Document
from src.module.document_ingestion_pipeline import DocumentIngestionPipeline
from src.module.retrieval_pipeline import RetrievalPipeline
from src.module.response_generation_pipeline import ResponseGenerationPipeline
from src.module.rag_pipeline import RAGPipeline
from src.utils import load_faqs
from src.constants import DATA_FOLDER, FAQ_FOLDER, HF_EMBEDDING_MODEL, PROMPT_TEMPLATE

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="Initializing RAG Engine...")
def get_rag_pipeline():
    """Initializes and caches the RAG pipeline. Only run once when the app starts"""
    try:
        # Load and ingest FAQs
        faq_path = os.path.join(DATA_FOLDER, FAQ_FOLDER)
        faqs = [Document(content=faq) for faq in load_faqs(faq_path)]

        vector_store = DocumentIngestionPipeline(
            hf_embedding_model=HF_EMBEDDING_MODEL
        ).create_vector_store(faqs)

        retrieval_pipeline = RetrievalPipeline(
            document_store=vector_store, hf_embedding_model=HF_EMBEDDING_MODEL, top_k=3
        )

        resp_gen_pipeline = ResponseGenerationPipeline(prompt_template=PROMPT_TEMPLATE)

        return RAGPipeline(retrieval_pipeline, resp_gen_pipeline)
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None


rag_pipeline = get_rag_pipeline()

st.title("Chatbot")
st.caption("Salam sejahtera, bagaimana saya boleh membantu anda?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_ques := st.chat_input("Masukkan soalan anda..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_ques})
    with st.chat_message("user"):
        st.markdown(user_ques)

    # Answer user query
    if rag_pipeline:
        with st.chat_message("assistant"):
            with st.spinner("Sedang fikir..."):
                output = rag_pipeline.answer(user_ques)

                if output.get("status") == "error":
                    st.error(f"Error: {output['error_message']}")
                else:
                    answer = output["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
    else:
        st.error("Sistem tidak berfungsi.")
