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
    # All internship details from CSV
    required_skills: Optional[str] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    stipend: Optional[float] = None
    remote: Optional[int] = None
    capacity: Optional[int] = None
    org_pref_govt: Optional[int] = None
    ministry: Optional[str] = None
    state: Optional[str] = None
    csr_underprivileged_pct: Optional[float] = None
    description: Optional[str] = None
    job_text: Optional[str] = None

class RecsResponse(BaseModel):
    student_id: str
    recommendations: List[RecItem]
