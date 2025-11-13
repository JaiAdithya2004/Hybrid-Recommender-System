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
    # Additional internship details
    stipend: Optional[float] = None
    remote: Optional[int] = None
    state: Optional[str] = None
    description: Optional[str] = None
    required_skills: Optional[str] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    capacity: Optional[int] = None

class RecsResponse(BaseModel):
    student_id: str
    recommendations: List[RecItem]
