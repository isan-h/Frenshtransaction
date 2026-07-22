import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import API_TITLE, API_DESCRIPTION, API_VERSION
from api.logging_config import setup_logging
from api.model_loader import load_models
from api.exceptions import (
    ModelNotLoadedError,
    PredictionError,
    model_not_loaded_handler,
    prediction_error_handler,
    unhandled_exception_handler,
)
from api.routers import predict as predict_router

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading models...")
    app.state.model_bundle = load_models()
    logger.info("Startup complete -- ready to accept requests.")

    yield 

    logger.info("Shutting down.")
    app.state.model_bundle = None


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.include_router(predict_router.router)

app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
app.add_exception_handler(PredictionError, prediction_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.get("/")
async def root():
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
    }
