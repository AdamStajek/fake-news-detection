from pydantic import BaseModel, Field


class DetectorModel(BaseModel):
    """A model for detecting fake news."""

    label: str = Field(description="Predicted label for the statement")
    explanation: str = Field(description="Explanation for the prediction")
