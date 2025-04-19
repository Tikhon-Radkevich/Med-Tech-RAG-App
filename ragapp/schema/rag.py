from typing import Optional
from pydantic import BaseModel, Field

from ragapp.device import DeviceEnum


class QueryCheck(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning behind query classification."
    )
    check: Optional[bool] = Field(
        default=False,
        description="Whether the query covers technical medical device topic.",
    )


class DeviceClassification(BaseModel):
    """Extracted device name classification result."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    device: Optional[DeviceEnum | None] = Field(
        default=None, description="Identified device name, or `None` if unclear."
    )


class ParaphrasedQuery(BaseModel):
    paraphrased_query: str = Field(description="Paraphrased query.")


class RelevantDocumentReference(BaseModel):
    """Reference document details."""

    title: str = Field(
        description="Title of the document, including file extension(e.g. `some_title.pdf`)."
    )
    pages: list[int] = Field(
        description="Relevant pages (do not duplicate same pages)."
    )


class RelevantDocumentSet(BaseModel):
    relevant_sources: list[RelevantDocumentReference] = Field(
        description=(
            "The ONLY documents and pages RELEVANT to the question. "
            "Empty list if no context or no relevant context is provided. "
            "Skip irrelevant sources."
        )
    )


class GraphExecutionError(BaseModel):
    message: str = Field(description="Error message.")
    stage: str = Field(description="Error stage.")
