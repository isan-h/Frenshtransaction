import logging

from fastapi import APIRouter, Request

from api.schemas import TransactionRequest, TransactionResponse
from api.exceptions import ModelNotLoadedError, PredictionError
from api.predictor import predict

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=TransactionResponse)
async def predict_transaction(payload: TransactionRequest, request: Request):
    bundle = request.app.state.model_bundle
    if bundle is None:
        raise ModelNotLoadedError()

    try:
        result = predict(payload.description, bundle)
    except Exception as exc:
        raise PredictionError(str(exc)) from exc

    return TransactionResponse(**result)


@router.get("/health")
async def health_check(request: Request):
    bundle = request.app.state.model_bundle
    is_ready = bundle is not None
    return {
        "status": "ready" if is_ready else "not_ready",
        "model_loaded": is_ready,
    }
