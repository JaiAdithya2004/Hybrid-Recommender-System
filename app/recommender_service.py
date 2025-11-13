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
    
    # Merge with internship details to get additional fields
    if not internships_df.empty and not sub.empty:
        # Merge recommendations with internship details
        merged = sub.merge(
            internships_df[['internship_id', 'stipend', 'remote', 'state', 'description', 
                          'required_skills', 'min_age', 'max_age', 'capacity']],
            on='internship_id',
            how='left'
        )
        # Convert to dict list with all fields
        result = merged[['student_id', 'internship_id', 'title', 'domain', 'score', 'rank',
                        'stipend', 'remote', 'state', 'description', 'required_skills',
                        'min_age', 'max_age', 'capacity']].to_dict(orient='records')
        
        # Handle NaN values - convert to None for JSON serialization
        for rec in result:
            for key, value in rec.items():
                if pd.isna(value):
                    rec[key] = None
            # Convert remote to int if not None
            if rec.get('remote') is not None:
                rec['remote'] = int(rec['remote'])
            # Convert numeric fields
            if rec.get('stipend') is not None:
                rec['stipend'] = float(rec['stipend'])
            if rec.get('min_age') is not None:
                rec['min_age'] = int(rec['min_age'])
            if rec.get('max_age') is not None:
                rec['max_age'] = int(rec['max_age'])
            if rec.get('capacity') is not None:
                rec['capacity'] = int(rec['capacity'])
        
        return result
    else:
        # Fallback to basic fields if merge fails
        return sub[['student_id','internship_id','title','domain','score','rank']].to_dict(orient='records')

def get_all_students():
    if students_df is not None and not students_df.empty:
        return students_df['student_id'].astype(str).tolist()
    return []
