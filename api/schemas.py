from pydantic import BaseModel, Field, field_validator
from api.config import MIN_DESCRIPTION_LENGTH, MAX_DESCRIPTION_LENGTH


class TransactionRequest(BaseModel):

    description: str = Field(
        ...,
        min_length=MIN_DESCRIPTION_LENGTH,
        max_length=MAX_DESCRIPTION_LENGTH,
        description="Raw transaction description, e.g. 'CB LIDL 4658 CERGY'",
        examples=["CB LIDL 4658 CERGY"],
    )

    @field_validator("description")
    @classmethod
    def description_must_not_be_only_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("description cannot be empty or only whitespace")
        return v


class TransactionResponse(BaseModel):
    description: str
    category: str = Field(description="Full category, e.g. 'Food & Drink / Groceries'")
    primary_category: str
    detailed_category: str
    category_confidence: float | None = Field(
        description="Model's confidence in the category prediction (0-1), if available"
    )
    merchant: str
    merchant_confidence: float | None = Field(
        description="Model's confidence in the merchant prediction (0-1), if available"
    )


class ErrorResponse(BaseModel):

    error: str
    detail: str
