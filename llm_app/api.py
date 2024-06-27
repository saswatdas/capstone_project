import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any
from typing import Union

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder


from llm_app import __version__, schemas
from config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(name=settings.PROJECT_NAME, api_version=__version__)

    return health.dict()


@api_router.post(
    "/retrieve_result", response_model=schemas.PredictResults, status_code=200
)
async def predict(input_data: str):

    results = "Testing.... "

    response = schemas.Health(llm_response=results, api_version=__version__)

    return response
