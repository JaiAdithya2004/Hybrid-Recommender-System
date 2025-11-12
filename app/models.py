# app/models.py
from sqlalchemy import Column, Integer, String, Float, Text
from app.db import Base

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, unique=True, index=True, nullable=False)
    domain = Column(String, nullable=True)
    state = Column(String, nullable=True)
    rural = Column(Integer, nullable=True)
    female = Column(Integer, nullable=True)
    github = Column(String, nullable=True)

class Internship(Base):
    __tablename__ = "internships"
    id = Column(Integer, primary_key=True, index=True)
    internship_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=True)
    domain = Column(String, nullable=True)
    stipend = Column(Float, nullable=True)
    capacity = Column(Integer, nullable=True)

class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    internship_id = Column(String)
    title = Column(String)
    domain = Column(String)
    score = Column(Float)
    rank = Column(Integer)
