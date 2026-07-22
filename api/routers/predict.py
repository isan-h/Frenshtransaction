"""
routers/predict.py
--------------------
The HTTP layer. Deliberately thin -- every route here just validates
via the request schema (automatic), calls predictor.predict(), and
returns. All the actual logic lives in predictor.py; this file's only
job is translating between "HTTP request" and "plain function call."
"""

import logging

from fastapi import APIRouter, Request

from api.schemas import TransactionRequest, TransactionResponse
from api.exceptions import ModelNotLoadedError, PredictionError
from api.predictor import predict

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=TransactionResponse)
async def predict_transaction(payload: TransactionRequest, request: Request):
    """
    Classify one transaction description into a category and merchant.

    payload is ALREADY validated by Pydantic by the time this function
    runs -- empty descriptions, wrong types, or overly long strings
    never reach this code at all; FastAPI rejects them automatically
    with a 422 response.
    """
    bundle = request.app.state.model_bundle
    if bundle is None:
        raise ModelNotLoadedError()

    try:
        result = predict(payload.description, bundle)
    except Exception as exc:
        # Anything that goes wrong INSIDE prediction (not validation,
        # which already happened) is treated as a prediction failure,
        # logged with the original exception, and turned into a clean
        # 500 by the handler in exceptions.py -- never a raw traceback.
        raise PredictionError(str(exc)) from exc

    return TransactionResponse(**result)


@router.get("/health")
async def health_check(request: Request):
    """Simple readiness check -- confirms the model is actually loaded,
    not just that the process is running. Useful for load balancers /
    container orchestration to know when the service can accept traffic."""
    bundle = request.app.state.model_bundle
    is_ready = bundle is not None
    return {
        "status": "ready" if is_ready else "not_ready",
        "model_loaded": is_ready,
    }
