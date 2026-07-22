"""
exceptions.py
-------------
Custom exceptions for this API, plus the handlers that convert them
into proper HTTP responses. Centralizing this here means every error
case returns the SAME response shape (see schemas.ErrorResponse),
instead of each route inventing its own error format ad hoc.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from api.schemas import ErrorResponse

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    """Raised if a prediction is attempted before startup finished
    loading the models -- should be effectively unreachable in normal
    operation (FastAPI won't accept requests until startup completes),
    but guards against it explicitly rather than trusting that."""
    pass


class PredictionError(Exception):
    """Wraps any unexpected failure during the actual prediction step
    (e.g. a malformed feature matrix) so it can be logged with context
    and turned into a clean 500 response, rather than a raw traceback."""
    pass


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    logger.error(f"Model not loaded when request hit {request.url.path}")
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="service_unavailable",
            detail="The model is not ready yet. Try again shortly.",
        ).model_dump(),
    )


async def prediction_error_handler(request: Request, exc: PredictionError):
    logger.error(f"Prediction failed on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="prediction_failed",
            detail="Something went wrong while generating a prediction.",
        ).model_dump(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last-resort catch-all -- logs the FULL exception server-side for
    debugging, but returns a generic message to the caller. Never leak
    internal stack traces or file paths to an external API consumer."""
    logger.exception(f"Unhandled exception on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred.",
        ).model_dump(),
    )
