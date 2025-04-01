from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from ragapp.prompts.prompts import (
    device_extractor_system_prompt_formatted,
    retriever_system_prompt,
    retriever_user_prompt,
)


class DeviceEnum(str, Enum):
    lifepak_15 = "lifepak 15"
    lifepak_20 = "lifepak 20"
    mizuho_6800 = "mizuho 6800"
    philips_m3002 = "philips m3002"
    defib_misc = "defib misc"
    philips_m3015 = "philips m3015"
    philips_m4841 = "philips m4841"
    philips_v60_vent = "philips v60 vent"


class DeviceClassification(BaseModel):
    """Extracted device classification result."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    device: Optional[DeviceEnum] = Field(
        description="Identified device name, or `None` if unclear."
    )


class SourceDocument(BaseModel):
    """Reference document details."""

    title: str = Field(description="Title of the document.")
    pages: list[int] = Field(description="Relevant pages.")


class RetrievalResult(BaseModel):
    """Retrieved information for answering a query."""

    sources: list[SourceDocument] = Field(
        description="Documents and pages, relevant to the query. Empty list if no context provided."
    )
    answer: str = Field(description="Generated answer to the question, if context provided.")


class VectorSearchParams(BaseModel):
    """Parameters for vector-based retrieval."""

    k: int = Field(default=5, description="Number of documents to retrieve.")
    filter: dict[str, str] = Field(default=None, description="Device filter.")


class QueryState(BaseModel):
    """State representation for the agent's decision flow."""

    question: str = Field(description="User query.")
    context_documents: list[Document] = Field(
        default=None, description="Retrieved documents."
    )
    device_classification: DeviceClassification = Field(
        default=None, description="Device classification result."
    )
    retrieval_result: RetrievalResult = Field(
        default=None, description="Final retrieval response."
    )
    search_params: VectorSearchParams = Field(
        default=VectorSearchParams(), description="Search parameters for retrieval."
    )


class MedTechAgent:
    def __init__(
        self, vector_store: VectorStore, rag_model_name: str, device_model_name: str
    ):
        self.rag_model_name = rag_model_name
        self.device_model_name = device_model_name
        self.vector_store = vector_store

        self.device_classifier = self._initialize_device_classifier()
        self.answer_generator = self._initialize_answer_generator()

    def compile(self) -> CompiledStateGraph:
        agent_graph = StateGraph(QueryState).add_node(self._classify_device)
        agent_graph = agent_graph.add_node(
            self._retrieve_documents.__name__, self._retrieve_documents
        )
        agent_graph = agent_graph.add_edge(
            START, end_key=self._classify_device.__name__
        )
        agent_graph = agent_graph.compile()
        return agent_graph

    def _classify_device(self, state: QueryState):
        classification = self.device_classifier.invoke(
            [
                {"role": "system", "content": device_extractor_system_prompt_formatted},
                {"role": "user", "content": state.question},
            ]
        )
        update = dict(device_classification=classification)

        if classification.device in list(DeviceEnum):
            search_params = dict(
                k=state.search_params.k,
                filter=dict(device=classification.device),
            )
            update.update(dict(search_params=VectorSearchParams(**search_params)))
            goto = self._retrieve_documents.__name__

        elif classification.device is None:
            goto = END
        else:
            raise RuntimeError(
                f"Invalid device classification: {classification.device}\n"
                f"Reasoning: {classification.reasoning}"
            )

        return Command(goto=goto, update=update)

    def _retrieve_documents(self, state: QueryState):
        print(state.search_params.model_dump())
        retrieved_docs = self.vector_store.similarity_search(
            state.question, **state.search_params.model_dump()
        )
        formatted_docs = self._format_documents(retrieved_docs)
        response = self.answer_generator.invoke(
            [
                {"role": "system", "content": retriever_system_prompt},
                {
                    "role": "user",
                    "content": retriever_user_prompt.format(
                        question=state.question, context=formatted_docs
                    ),
                },
            ]
        )
        update = {"retrieval_result": response, "context_documents": retrieved_docs}
        return Command(goto=END, update=update)

    def _initialize_answer_generator(self):
        model = init_chat_model(model=self.rag_model_name, temperature=0)
        return model.with_structured_output(RetrievalResult)

    def _initialize_device_classifier(self):
        model = init_chat_model(model=self.device_model_name, temperature=0)
        return model.with_structured_output(DeviceClassification)

    @staticmethod
    def _format_documents(docs) -> str:
        return "\n\n".join(
            f"Source: {doc.metadata['pdf_title']}\n{doc.page_content}" for doc in docs
        )
