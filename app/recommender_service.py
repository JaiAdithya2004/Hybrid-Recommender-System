# app/recommender_service.py
import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_candidate_dirs = [
    ROOT / "outputs_recommender_v2",
    ROOT / "notebook" / "outputs_recommender_v2",
]
OUT_DIR = next((path for path in _candidate_dirs if path.exists()), _candidate_dirs[0])

# Load precomputed recommendations CSV (must exist)
RECS_CSV = OUT_DIR / "recommendations.csv"
STUDENTS_CSV = OUT_DIR / "students_synthetic.csv"
INTERNS_CSV = OUT_DIR / "internships_synthetic.csv"

# load on import (fast reads)
if RECS_CSV.exists():
    recs_df = pd.read_csv(RECS_CSV)
else:
    recs_df = pd.DataFrame(columns=['student_idx','student_id','intern_idx','internship_id','title','domain','score','rank'])

if STUDENTS_CSV.exists():
    students_df = pd.read_csv(STUDENTS_CSV)
else:
    students_df = pd.DataFrame()

if INTERNS_CSV.exists():
    internships_df = pd.read_csv(INTERNS_CSV)
else:
    internships_df = pd.DataFrame()

def recommend_for_student(student_id: str, top_k: int = 10):
    if student_id not in recs_df['student_id'].unique():
        # fallback: return empty list
        return []
    sub = recs_df[recs_df['student_id'] == student_id].sort_values('rank').head(top_k)
    # convert to dict list
    return sub[['student_id','internship_id','title','domain','score','rank']].to_dict(orient='records')

def get_all_students():
    if students_df is not None and not students_df.empty:
        return students_df['student_id'].astype(str).tolist()
    return []
