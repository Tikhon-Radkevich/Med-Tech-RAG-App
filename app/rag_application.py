import logging
from dotenv import load_dotenv

import torch

# https://github.com/VikParuchuri/marker/issues/442#issuecomment-2636393925
torch.classes.__path__ = []

import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ragapp.rag.med_agent_graph import MedTechAgent, AgentQueryState
from ragapp.schema.rag import RelevantDocumentReference
from constants import (
    TOGETHER_META_LLAMA_70B_FREE,
    GROQ_LLAMA_70B,
    GROQ_QWEN_32B,
    GROQ_GEMMA_9B,
    GROQ_LLAMA_SCOUT_17B,
    EMBEDDINGS_MODEL_NAME,
    LP_DOCLING_CACHE_DIR,
    LP_PLUMBER_CACHE_DIR,
    LP_DOCLING_COLLECTION_NAME,
    LP_PLUMBER_COLLECTION_NAME,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
)
logger.addHandler(console_handler)

embeddings_model_name = EMBEDDINGS_MODEL_NAME
collection_name = LP_DOCLING_COLLECTION_NAME
persist_directory = LP_DOCLING_CACHE_DIR

st.info(f"Embeddings: {embeddings_model_name}")
st.info(f"Index: CHROMA: {collection_name}")


def format_sources_markdown(sources: list[RelevantDocumentReference]):
    markdown = "---\n\n"
    markdown += "### Sources\n\n"
    for source in sources:
        title = source.title
        pages = ", ".join(map(str, source.pages))
        markdown += f"- **{title}**  \n\t- pages: {pages}\n"
    return markdown


@st.cache_resource
def load_agent() -> MedTechAgent:
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    agent = MedTechAgent(
        vector_storage=vector_store,
        k=5,
        query_check_model=GROQ_QWEN_32B,
        device_classifier_model=GROQ_QWEN_32B,
        paraphraser_model=GROQ_LLAMA_SCOUT_17B,
        relevance_selector_model=TOGETHER_META_LLAMA_70B_FREE,
        answer_generator_model=GROQ_LLAMA_70B,
        # answering_model="groq:llama-3.2-90b-vision-preview",
        error_handler_model=GROQ_LLAMA_70B,
    )
    return agent


def main():
    agent = load_agent()

    st.title("🦜🔗 RAG App")
    with st.form("chat_form"):
        text = st.text_area(
            "Enter text:",
            "How do I troubleshoot low volume on Lifepak 20",
            # "How to perform leakage current test on LP 15?"
        )
        submitted = st.form_submit_button("Submit")
        if not submitted:
            return
        if not text:
            st.info("No text entered!")
            return

        try:
            response = agent.run(question=text)
            res = response.final_response

            if response.paraphrased_question is not None:
                st.info(response.paraphrased_question)

            if response.filtered_relevant_documents is not None:
                st.markdown(res, unsafe_allow_html=True)
                sources = format_sources_markdown(
                    response.filtered_relevant_documents.relevant_sources
                )
                st.markdown(sources, unsafe_allow_html=True)
            else:
                st.info(res)
        except Exception as e:
            st.info(e)


if __name__ == "__main__":
    load_dotenv()
    main()
