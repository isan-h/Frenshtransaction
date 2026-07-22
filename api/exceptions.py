import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from api.schemas import ErrorResponse

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    pass


class PredictionError(Exception):
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
    logger.exception(f"Unhandled exception on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred.",
        ).model_dump(),
    )
