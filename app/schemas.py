# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class RecItem(BaseModel):
    student_id: str
    internship_id: str
    title: str
    domain: Optional[str]
    score: float
    rank: int

class RecsResponse(BaseModel):
    student_id: str
    recommendations: List[RecItem]
