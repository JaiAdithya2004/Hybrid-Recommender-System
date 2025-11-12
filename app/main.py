# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pathlib import Path
import os, pandas as pd

from app.db import SessionLocal, engine
from app import crud, recommender_service, schemas

# create tables
crud.create_tables(engine)

app = FastAPI(title="Hybrid Recommender API")

# CORS open for testing (later restrict to your frontend domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

ROOT = Path(__file__).resolve().parents[1]
_candidate_dirs = [
    ROOT / "outputs_recommender_v2",
    ROOT / "notebook" / "outputs_recommender_v2",
]
OUT_DIR = next((path for path in _candidate_dirs if path.exists()), _candidate_dirs[0])

@app.get("/")
def health():
    return {"status":"ok", "message":"Recommender API running"}

@app.post("/populate_db")
def populate_db(db: Session = Depends(get_db)):
    # Load CSVs and populate students, internships, recommendations into sqlite
    students_csv = OUT_DIR / "students_synthetic.csv"
    internships_csv = OUT_DIR / "internships_synthetic.csv"
    recs_csv = OUT_DIR / "recommendations.csv"

    if not students_csv.exists() or not internships_csv.exists():
        raise HTTPException(status_code=400, detail="students or internships CSV missing in outputs_recommender_v2")

    crud.load_students_from_csv(db, str(students_csv))
    crud.load_internships_from_csv(db, str(internships_csv))

    # optionally populate recommendations table
    if recs_csv.exists():
        df = pd.read_csv(str(recs_csv))
        crud.save_recommendations_from_df(db, df, reset=True)

    return {"status":"ok", "message":"DB populated from CSVs"}

@app.get("/students")
def list_students():
    s = recommender_service.get_all_students()
    return {"count": len(s), "students": s}

@app.get("/internships")
def list_internships():
    path = OUT_DIR / "internships_synthetic.csv"
    if path.exists():
        df = pd.read_csv(path)
        return {"count": len(df), "internships": df[['internship_id','title','domain']].to_dict(orient='records')}
    return {"count": 0, "internships": []}

@app.get("/recommend/{student_id}", response_model=schemas.RecsResponse)
def recommend(student_id: str, top_k: int = 10):
    recs = recommender_service.recommend_for_student(student_id, top_k=top_k)
    # convert to schema format
    return {"student_id": student_id, "recommendations": recs}

@app.post("/recommend_and_store/{student_id}")
def recommend_and_store(student_id: str, top_k: int = 10, db: Session = Depends(get_db)):
    recs = recommender_service.recommend_for_student(student_id, top_k=top_k)
    if not recs:
        return {"student_id": student_id, "recommendations_saved": 0}
    # convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(recs)
    # ensure expected columns: student_id, internship_id, title, domain, score, rank
    crud.save_recommendations_from_df(db, df, reset=False)
    return {"student_id": student_id, "recommendations_saved": len(df)}
