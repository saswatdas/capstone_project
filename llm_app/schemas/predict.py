from typing import Any, List, Optional

from pydantic import BaseModel


class PredictResults(BaseModel):
    version: str
    llm_response: Optional[str]
