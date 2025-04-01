from dotenv import load_dotenv

import torch

torch.classes.__path__ = []

import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph.state import CompiledStateGraph

from ragapp.rag.med_agent_graph import MedTechAgent, QueryState
from constants import (
    TOGETHER_META_LLAMA_70B_FREE,
    EMBEDDINGS_MODEL_NAME,
    INDEX_CACHE_DIR,
)


chat_mode_name = TOGETHER_META_LLAMA_70B_FREE
embeddings_model_name = EMBEDDINGS_MODEL_NAME
index_name = "LIFEPAK_index"
index_type = "CHROMA"

st.info(f"Chat Model: {chat_mode_name}")
st.info(f"Embeddings: {embeddings_model_name}")
st.info(f"Index: {index_type}")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def load_agent() -> CompiledStateGraph:
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vector_store = Chroma(
        collection_name=index_name,
        embedding_function=embeddings,
        persist_directory=INDEX_CACHE_DIR,
    )

    graph_agent = MedTechAgent(
        vector_store=vector_store,
        rag_model_name=chat_mode_name,
        device_model_name=chat_mode_name,
    ).compile()
    return graph_agent


def main():
    agent = load_agent()

    st.title("🦜🔗 RAG App")
    with st.form("chat_form"):
        text = st.text_area(
            "Enter text:",
            "How do I troubleshoot low volume on Lifepak 20?",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            if text:
                inputs = {"question": text, "k": 5}
                response = QueryState(**agent.invoke(inputs))

                if response.retrieval_result is not None:
                    res = response.retrieval_result.answer
                else:
                    res = response.device_classification.reasoning

                st.info(res)
            else:
                st.info("No text entered!")


if __name__ == "__main__":
    load_dotenv()
    main()
