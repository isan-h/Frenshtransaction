"""
model_loader.py
-----------------
Loads the feature pipeline, both label encoders, and both trained
models EXACTLY ONCE, and holds them in memory for the life of the
process. This is the module that makes "never retrain during
prediction" and "load model only once" actual guarantees, not just
intentions.

Design: a single ModelBundle instance, created once via load_models()
and called from main.py's startup lifespan. Not a global singleton
that loads on import -- explicit construction at startup means a
failure to load surfaces immediately and loudly, before the API
accepts a single request, rather than at some unpredictable later
moment.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import joblib

from api.config import (
    FEATURE_PIPELINE_PATH,
    CATEGORY_ENCODER_PATH,
    MERCHANT_ENCODER_PATH,
    CATEGORY_MODEL_PATH,
    MERCHANT_MODEL_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Everything a prediction needs, loaded once, held in memory.
    A dataclass instead of 5 loose global variables -- one object to
    pass around, one thing to check `is None`, one thing to mock in tests.
    """
    feature_pipeline: Any
    category_encoder: Any
    merchant_encoder: Any
    category_model: Any
    merchant_model: Any


def load_models() -> ModelBundle:
    """
    Loads every artifact from disk ONCE. Called from main.py's startup
    lifespan -- never called again during the life of the process.

    Raises FileNotFoundError (via joblib.load) if any artifact is
    missing -- this is intentional: we want the API to refuse to start
    rather than start "successfully" and then fail on the first request.
    """
    start = time.time()
    logger.info("Loading ML artifacts from disk (this happens ONCE, at startup)...")

    feature_pipeline = joblib.load(FEATURE_PIPELINE_PATH)
    category_encoder = joblib.load(CATEGORY_ENCODER_PATH)
    merchant_encoder = joblib.load(MERCHANT_ENCODER_PATH)
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    merchant_model = joblib.load(MERCHANT_MODEL_PATH)

    elapsed = time.time() - start
    logger.info(
        f"Loaded feature pipeline, 2 label encoders, and 2 models in {elapsed:.2f}s. "
        f"Category classes: {len(category_encoder.classes_)}, "
        f"Merchant classes: {len(merchant_encoder.classes_)}."
    )

    return ModelBundle(
        feature_pipeline=feature_pipeline,
        category_encoder=category_encoder,
        merchant_encoder=merchant_encoder,
        category_model=category_model,
        merchant_model=merchant_model,
    )
