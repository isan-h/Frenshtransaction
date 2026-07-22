"""
schemas.py
----------
Pydantic models define the API's data contract: exactly what a valid
request must look like, and exactly what a response will look like.
FastAPI uses these automatically to:
  1. Reject invalid requests BEFORE your code runs (empty description,
     wrong type, description too long) -- with a clear 422 error
     explaining exactly what was wrong.
  2. Generate the interactive docs at /docs for free.
  3. Serialize responses consistently -- no risk of a typo'd key name
     in one response but not another.
"""

from pydantic import BaseModel, Field, field_validator

from api.config import MIN_DESCRIPTION_LENGTH, MAX_DESCRIPTION_LENGTH


class TransactionRequest(BaseModel):
    """What the caller must send. min_length/max_length reject obviously
    bad input (empty string, or someone pasting a whole document) before
    it ever reaches the model."""

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
        """min_length counts whitespace characters -- "   " (3 spaces)
        would pass a bare min_length=3 check despite being empty in any
        meaningful sense. This closes that specific gap explicitly."""
        if not v.strip():
            raise ValueError("description cannot be empty or only whitespace")
        return v


class TransactionResponse(BaseModel):
    """What every successful prediction returns. Both category and
    merchant are ALWAYS present -- there's no partial-success state
    for one head failing while the other succeeds, since they share
    the same input features and either both run or neither does."""

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
    """Consistent shape for every error the API returns, regardless of
    which exception triggered it -- callers can rely on `error` and
    `detail` always being present, rather than guessing the shape of
    each possible failure."""

    error: str
    detail: str
