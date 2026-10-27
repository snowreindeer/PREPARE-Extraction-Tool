from typing import List, Optional
from pydantic import BaseModel, Field

# =====================================
# Data types
# =====================================

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: Optional[float]

# =====================================
# LitServe interface
# =====================================

class NERRequest(BaseModel):
    medical_text: str
    labels: list[str] | dict[str, str] | None = None

# =====================================
# API interface
# =====================================

class LabelsInput(BaseModel):
    labels: List[str] | None = None



# =====================================
# Clinical processing data contracts
# =====================================

class Token(BaseModel):
    """Token span with offsets relative to the original record text."""

    text: str
    norm: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class SentenceSpan(BaseModel):
    """Sentence character and token offsets."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    token_start: int = Field(ge=0)
    token_end: int = Field(ge=0)
