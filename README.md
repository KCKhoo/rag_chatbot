# Retrieval-Augmented Generation (RAG) Chatbot

## Goals
1. Develop a RAG chatbot to provide accurate answers to user queries using FAQs
2. Create a Streamlit UI to interact with the chatbot in real-time. Link: https://ragmalaychatbot.streamlit.app/

## Tech Stack
- LLM: Google Gemini
- RAG Framework: Haystack
- Vector Database: Qdrant
- Embedding Model: `BAAI/bge-m3`

## High Level System Architecture

1. Document Ingestion: Document Loading $\rightarrow$ Chunking $\rightarrow$ Embedding $\rightarrow$ Vector Database
2. Retrieval: User Query $\rightarrow$ Semantic search for relevant context using embeddings
3. Response Generation: User Query + Context + Prompt $\rightarrow$ LLM Response

## Installation (Conda)

```
# Create the environment
conda create -n <YOUR_ENV_NAME> python=3.13

# Activate the environment
conda activate <YOUR_ENV_NAME>

# Install dependencies
pip install -r requirements.txt
```

## Configuration
Create a .env file in the root directory and add your API keys:
```
HF_TOKEN=your_key_here
GEMINI_API_KEY=your_key_here
```

## Steps for Launching Streamlit Locally

1. Ensure the .env file is configured
2. Launch Streamlit server by running the following terminal command from the root directory:
    ```
    streamlit run app.py
    ```
3. Click the local URL (usually http://localhost:8501) shown in the terminal to open the interface.

## Remarks
1. The chatbot can only provide answer that can be found in the FAQs.

2. If the answer is not found in the FAQ, the model is instructed to reply "No Information"
