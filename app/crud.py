# app/crud.py
from sqlalchemy.orm import Session
from app import models
import pandas as pd

def create_tables(engine):
    models.Base.metadata.create_all(bind=engine)

def load_students_from_csv(db: Session, csv_path: str):
    df = pd.read_csv(csv_path)
    # basic upsert: clear table then bulk insert (simple approach)
    db.query(models.Student).delete()
    for _, r in df.iterrows():
        s = models.Student(
            student_id=str(r.get('student_id')),
            domain=r.get('domain'),
            state=r.get('state'),
            rural=int(r.get('rural', 0)) if not pd.isna(r.get('rural', 0)) else None,
            female=int(r.get('female', 0)) if not pd.isna(r.get('female', 0)) else None,
            github=r.get('github', '')
        )
        db.add(s)
    db.commit()

def load_internships_from_csv(db: Session, csv_path: str):
    df = pd.read_csv(csv_path)
    db.query(models.Internship).delete()
    for _, r in df.iterrows():
        it = models.Internship(
            internship_id=str(r.get('internship_id')),
            title=r.get('title'),
            domain=r.get('domain'),
            stipend=float(r.get('stipend')) if not pd.isna(r.get('stipend')) else None,
            capacity=int(r.get('capacity')) if not pd.isna(r.get('capacity')) else None
        )
        db.add(it)
    db.commit()

def save_recommendations_from_df(db: Session, recs_df: pd.DataFrame, reset=True):
    if reset:
        db.query(models.Recommendation).delete()
        db.commit()
    for _, r in recs_df.iterrows():
        rec = models.Recommendation(
            student_id=str(r['student_id']),
            internship_id=str(r['internship_id']),
            title=r.get('title'),
            domain=r.get('domain'),
            score=float(r.get('score', 0.0)),
            rank=int(r.get('rank', 0))
        )
        db.add(rec)
    db.commit()

def get_recommendations(db: Session, student_id: str, top_k: int = 10):
    return db.query(models.Recommendation).filter(models.Recommendation.student_id==student_id).order_by(models.Recommendation.rank).limit(top_k).all()
