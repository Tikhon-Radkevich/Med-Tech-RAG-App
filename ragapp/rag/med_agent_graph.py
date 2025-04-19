import logging
import json
from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum

from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain_chroma import Chroma

from ragapp.device import DeviceEnum
from ragapp.schema.rag import (
    QueryCheck,
    DeviceClassification,
    ParaphrasedQuery,
    RelevantDocumentSet,
    GraphExecutionError,
)

from ragapp.prompts import (
    query_check_system_prompt,
    query_check_user_prompt,
    device_extractor_system_prompt_formatted,
    rephrase_system_prompt,
    rephrase_user_prompt,
    relevant_context_selector_system_prompt,
    relevant_context_selector_user_prompt,
    query_answer_system_prompt,
    query_answer_user_prompt,
    error_handing_system_prompt,
    error_handing_user_prompt,
)


logger = logging.getLogger(__name__)


class AgentQueryState(BaseModel):
    """State representation for the agent's decision flow."""

    original_question: str = Field(default=None, description="User query.")
    paraphrased_question: str = Field(default=None, description="Rephrased user query.")
    final_response: str = Field(default=None, description="Answer to the query.")
    query_check: QueryCheck = Field(default=None, description="Query check.")
    device_classification: DeviceClassification = Field(
        default=None, description="Device classification result."
    )
    retrieved_documents: list[Document] = Field(
        default=None, description="Retrieved documents."
    )
    filtered_relevant_documents: RelevantDocumentSet = Field(
        default=None, description="Retrieved documents."
    )
    execution_error: GraphExecutionError = Field(default=None, description="Error Log")
    state_kwargs: Optional[Any] = Field(
        default=None, description="Any kwargs to be added in response."
    )


class StateAttrNames(str, Enum):
    """
    Attribute names of AgentQueryState, used for updating state values within the agent graph.
    Values must exactly match the field names in AgentQueryState to ensure compatibility.
    """

    original_question = "original_question"
    paraphrased_question = "paraphrased_question"
    final_response = "final_response"
    query_check = "query_check"
    device_classification = "device_classification"
    retrieved_documents = "retrieved_documents"
    filtered_relevant_documents = "filtered_relevant_documents"
    execution_error = "execution_error"
    state_kwargs = "state_kwargs"


class MedTechAgent:
    """
    0. Check Query.
    1. Device Classification.
    2. Query Rephrase.
    3. Retrieve Documents.
    4. Select Relevant Documents.
    5. Answer to the query, based on relevant documents.
    """

    def __init__(
        self,
        vector_storage: Chroma,
        query_check_model: str,
        device_classifier_model: str,
        paraphraser_model: str,
        relevance_selector_model: str,
        answer_generator_model: str,
        error_handler_model: str,
        k: int = 5,
        max_doc_length_preview: int = 50,
    ):
        self.vector_storage = vector_storage
        self.k = k
        self._max_doc_length_preview = max_doc_length_preview

        self._answer_generator = self._initialize_model(answer_generator_model)
        self._error_processor = self._initialize_model(error_handler_model)
        self._query_checker = self._initialize_model(query_check_model, QueryCheck)
        self._query_paraphraser = self._initialize_model(
            paraphraser_model, ParaphrasedQuery
        )
        self._device_classifier = self._initialize_model(
            device_classifier_model, DeviceClassification
        )
        self._relevant_doc_selector = self._initialize_model(
            relevance_selector_model, RelevantDocumentSet
        )

        self._agent = self._compile()

    def run(self, question: str, **kwargs) -> AgentQueryState:
        """
        Run query through graph agent.

        :param question: query question
        :param kwargs: any kwargs to be added in response.
        :return: `QueryState`
        """

        invoke_kwargs = {
            StateAttrNames.original_question: question,
            StateAttrNames.state_kwargs: kwargs,
        }
        return AgentQueryState(**self._agent.invoke(invoke_kwargs))

    def retrieve(self, query: str, k: int, device: str) -> list[Document]:
        return self.vector_storage.similarity_search(
            query, k=k, filter={"device": device}
        )

    def _query_check(self, state: AgentQueryState):
        """Validate that the query is specifically related to
        technical issues, usage, or maintenance of a medical device."""

        update = dict()
        try:
            user_prompt = query_check_user_prompt.format(
                query=state.original_question,
            )
            response: QueryCheck = self._query_checker.invoke(
                self._form_invoke_input(query_check_system_prompt, user_prompt)
            )
            update[StateAttrNames.query_check] = response
            if not response.check:
                raise RuntimeError("Query check failed")

            goto = self._classify_device.__name__

        except Exception as e:
            logger.exception("Error during query checking.")
            logger.exception(e)
            update[StateAttrNames.execution_error] = GraphExecutionError(
                message=str(e),
                stage="Query Check",
            )
            goto = self._handle_graph_error.__name__

        return Command(goto=goto, update=update)

    def _classify_device(self, state: AgentQueryState):
        update = dict()
        try:
            classification: DeviceClassification = self._device_classifier.invoke(
                self._form_invoke_input(
                    device_extractor_system_prompt_formatted, state.original_question
                )
            )
            update[StateAttrNames.device_classification] = classification

            if classification is None:
                raise RuntimeError("Device classification failed.")

            if classification.device not in list(DeviceEnum):
                raise RuntimeError(
                    f"Invalid device classification: {classification.device}\n"
                )
            goto = self._generate_paraphrased_query.__name__

        except Exception as e:
            logger.exception("Error during device classification.")
            logger.exception(e)
            update[StateAttrNames.execution_error] = GraphExecutionError(
                message=str(e),
                stage="Device Classification",
            )
            goto = self._handle_graph_error.__name__

        return Command(goto=goto, update=update)

    def _generate_paraphrased_query(self, state: AgentQueryState):
        update = dict()
        try:
            user_prompt = rephrase_user_prompt.format(
                query=state.original_question,
                device=state.device_classification.device,
            )
            response: ParaphrasedQuery = self._query_paraphraser.invoke(
                self._form_invoke_input(rephrase_system_prompt, user_prompt)
            )
            update[StateAttrNames.paraphrased_question] = response.paraphrased_query
            goto = self._retrieve_context_documents.__name__
        except Exception as e:
            logger.exception("Error during query response.")
            logger.exception(e)
            update[StateAttrNames.execution_error] = GraphExecutionError(
                message=str(e),
                stage="Query Paraphrasing",
            )
            goto = self._handle_graph_error.__name__

        return Command(goto=goto, update=update)

    def _retrieve_context_documents(self, state: AgentQueryState):
        update = dict()
        try:
            update[StateAttrNames.retrieved_documents] = self.retrieve(
                state.paraphrased_question,
                k=self.k,
                device=str(state.device_classification.device.value),
            )
            goto = self._filter_relevant_documents.__name__

        except Exception as e:
            logger.exception("Error during retrieving documents.")
            logger.exception(e)
            update[StateAttrNames.execution_error] = GraphExecutionError(
                message=str(e),
                stage="Document Retrieval",
            )
            goto = self._handle_graph_error.__name__

        return Command(goto=goto, update=update)

    def _filter_relevant_documents(self, state: AgentQueryState):
        update = dict()
        try:
            formatted_docs = self._format_documents(state.retrieved_documents)
            user_prompt = relevant_context_selector_user_prompt.format(
                query=state.paraphrased_question,
                docs=formatted_docs,
            )
            relevant_documents: RelevantDocumentSet = (
                self._relevant_doc_selector.invoke(
                    self._form_invoke_input(
                        relevant_context_selector_system_prompt, user_prompt
                    )
                )
            )
            update[StateAttrNames.filtered_relevant_documents] = relevant_documents
            goto = self._generate_final_answer.__name__

        except Exception as e:
            logger.exception("Error during query response.")
            logger.exception(e)
            update[StateAttrNames.execution_error] = GraphExecutionError(
                message=str(e),
                stage="Relevant Documents Selection",
            )
            goto = self._handle_graph_error.__name__

        return Command(goto=goto, update=update)

    def _generate_final_answer(self, state: AgentQueryState):
        update = dict()
        try:
            relevant_docs_list = self._filter_relevant_docs(
                state.retrieved_documents, state.filtered_relevant_documents
            )
            formatted_docs = self._format_documents(relevant_docs_list)
            user_prompt = query_answer_user_prompt.format(
                query=state.paraphrased_question,
                context=formatted_docs,
            )
            response = self._answer_generator.invoke(
                self._form_invoke_input(query_answer_system_prompt, user_prompt)
            )
            update[StateAttrNames.final_response] = response.content
            goto = END

        except Exception as e:
            logger.exception("Error during query response.")
            logger.exception(e)
            update[StateAttrNames.execution_error] = GraphExecutionError(
                message=str(e),
                stage="Question Answering",
            )
            goto = self._handle_graph_error.__name__

        return Command(goto=goto, update=update)

    def _handle_graph_error(self, state: AgentQueryState):
        update = dict()
        try:
            context_docs_content = "No Context Documents Provided."
            if state.retrieved_documents is not None:
                context_docs_content = self._format_documents(
                    state.retrieved_documents, self._max_doc_length_preview
                )

            state_dict = state.model_dump(exclude={"context_documents"})
            state_dict["context_docs_content"] = context_docs_content
            state_dict_string = json.dumps(state_dict, indent=2, ensure_ascii=False)

            user_prompt = error_handing_user_prompt.format(
                stage=state.execution_error.stage,
                error_message=state.execution_error.message,
                state_dict=state_dict_string,
            )

            response = self._error_processor.invoke(
                self._form_invoke_input(error_handing_system_prompt, user_prompt)
            )
            update[StateAttrNames.final_response] = response.content
            goto = END

        except Exception as e:
            logger.exception("Error during error handling.")
            logger.exception(e)
            update[StateAttrNames.final_response] = "Something went wrong, try again."
            goto = END

        return Command(goto=goto, update=update)

    def _compile(self) -> CompiledStateGraph:
        agent_graph = StateGraph(AgentQueryState).add_node(self._query_check)
        agent_graph = agent_graph.add_node(
            self._classify_device.__name__, self._classify_device
        )
        agent_graph = agent_graph.add_node(
            self._generate_paraphrased_query.__name__, self._generate_paraphrased_query
        )
        agent_graph = agent_graph.add_node(
            self._retrieve_context_documents.__name__, self._retrieve_context_documents
        )
        agent_graph = agent_graph.add_node(
            self._filter_relevant_documents.__name__, self._filter_relevant_documents
        )
        agent_graph = agent_graph.add_node(
            self._generate_final_answer.__name__, self._generate_final_answer
        )
        agent_graph = agent_graph.add_node(
            self._handle_graph_error.__name__, self._handle_graph_error
        )
        agent_graph = agent_graph.add_edge(START, end_key=self._query_check.__name__)
        return agent_graph.compile()

    @staticmethod
    def _filter_relevant_docs(
        docs: list[Document], relevant: RelevantDocumentSet
    ) -> list[Document]:
        """
        Select relevant langchain documents based on provided relevant titles and pages.
        :param docs: langchain docs
        :param relevant: relevant doc titles and pages
        :return: relevant langchain docs
        """
        relevant_set = set(
            [
                (doc.title, page)
                for doc in relevant.relevant_sources
                for page in doc.pages
            ]
        )
        relevant_docs = [
            doc
            for doc in docs
            if (doc.metadata["pdf_title"], doc.metadata["page"]) in relevant_set
        ]
        return relevant_docs

    @staticmethod
    def _initialize_model(model_name: str, schema=None) -> Runnable:
        model = init_chat_model(model=model_name, temperature=0)
        if schema is not None:
            model = model.with_structured_output(schema)
        return model

    @staticmethod
    def _form_invoke_input(system_prompt: str, user_prompt: str):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _format_documents(docs: list[Document], max_content_length=None) -> str:
        def format_content(content: str) -> str:
            if max_content_length is not None:
                return content[:max_content_length] + "..."
            return content

        formated_documents = "\n\n".join(
            f"--- SOURCE {i} ---"
            f"\nFILE: {doc.metadata['pdf_title']}"
            f"\nPAGE: {doc.metadata['page']}"
            f"\n{format_content(doc.page_content)}"
            for i, doc in enumerate(docs, start=1)
        )
        if not formated_documents:
            formated_documents = "No Context Documents Provided."
        return formated_documents
