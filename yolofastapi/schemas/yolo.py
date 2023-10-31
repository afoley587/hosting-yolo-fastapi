from pydantic import BaseModel
from typing import Set


class ImageAnalysisResponse(BaseModel):
    id: int
    labels: Set[str]
