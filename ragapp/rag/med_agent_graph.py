from pydantic import BaseModel, Field
from typing import Optional, Any

from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_chroma import Chroma

from ragapp.device import DeviceEnum
from ragapp.prompts.prompts import (
    device_extractor_system_prompt_formatted,
    retriever_system_prompt,
    retriever_user_prompt,
)


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
        description="Documents and pages, relevant to the query. Empty list if no context provided. "
        "IMPORTANT: select only sources, that where used to form the answer. "
        "Skip not relevant sources."
    )
    answer: str = Field(
        description="Generated answer to the question, if context provided."
    )


class QueryState(BaseModel):
    """State representation for the agent's decision flow."""

    question: str = Field(description="User query.")
    k: int = Field(default=5, description="Number of documents to retrieve.")
    context_documents: list[Document] = Field(
        default=None, description="Retrieved documents."
    )
    device_classification: DeviceClassification = Field(
        default=None, description="Device classification result."
    )
    retrieval_result: RetrievalResult = Field(
        default=None, description="Final retrieval response."
    )
    state_kwargs: Optional[dict[Any, Any]] = Field(
        description="Any kwargs to be added in response."
    )


class MedTechAgent:
    def __init__(
        self, vector_storage: Chroma, rag_model_name: str, device_model_name: str
    ):
        self.rag_model_name = rag_model_name
        self.device_model_name = device_model_name
        self.vector_storage = vector_storage

        self.device_classifier = self._initialize_device_classifier()
        self.answer_generator = self._initialize_answer_generator()

        self.agent = self.compile()

    def run(self, question: str, k: int = 5, **kwargs) -> QueryState:
        """
        Run query thought graph aget.

        :param question: query question
        :param k: top k values to aggregate with retriever
        :param kwargs: any kwargs to be added in response.
        :return: `QueryState`
        """

        invoke_kwargs = dict(
            k=k,
            question=question,
            state_kwargs=kwargs,
        )
        return QueryState(**self.agent.invoke(invoke_kwargs))

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

    def retrieve(self, query: str, k: int, device: str) -> list[Document]:
        return self.vector_storage.similarity_search(
            query, k=k, filter={"device": device}
        )

    def _classify_device(self, state: QueryState):
        classification = self.device_classifier.invoke(
            [
                {"role": "system", "content": device_extractor_system_prompt_formatted},
                {"role": "user", "content": state.question},
            ]
        )
        update = dict(device_classification=classification)

        if classification.device in list(DeviceEnum):
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
        retrieved_docs = self.retrieve(
            state.question,
            k=state.k,
            device=str(state.device_classification.device.value),
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
            f"--- SOURCE {i} ---"
            f"\nFILE: {doc.metadata['pdf_title']}"
            f"\nPAGE: {doc.metadata['page']}"
            f"\n{doc.page_content}"
            for i, doc in enumerate(docs, start=1)
        )
