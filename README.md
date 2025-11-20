# 🎓 Hybrid AI-Powered Internship Recommendation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)](https://fastapi.tiangolo.com/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-red.svg)](https://www.sqlalchemy.org/)
[![Sentence-Transformers](https://img.shields.io/badge/SBERT-latest-orange.svg)](https://www.sbert.net/)


> **Production-ready recommendation system combining semantic matching (SBERT embeddings), collaborative filtering, and optimization algorithms. Features REST API, SQLite persistence, and comprehensive performance evaluation suite.**

### Access Here :   https://intern-match-ai-gold.vercel.app/

### 🎯 Key Highlights
- 🤖 **Hybrid Recommendation Engine**: Content-based (SBERT) + Collaborative Filtering + Optimization (OR-Tools)
- ⚡ **High-Performance API**: FastAPI backend serving precomputed recommendations with sub-100ms latency
- 📊 **Intelligent Matching**: Semantic similarity using sentence-transformers (all-mpnet-base-v2)
- 🗄️ **Dual Data Layer**: CSV-based serving + SQLite persistence with SQLAlchemy ORM
- 📈 **Evaluation Framework**: Built-in metrics for coverage, diversity, popularity bias (Gini), and allocation fairness
- 🔄 **Scalable Architecture**: Offline computation pipeline + lightweight serving layer

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Recommendation Pipeline](#-recommendation-pipeline)
- [Performance Evaluation](#-performance-evaluation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Deployment](#-deployment)


## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE (Notebook)                  │
│                                                                 │
│  Students Data ──┐                                              │
│  Internships ────┼──▶ Feature Engineering ──▶ SBERT Embeddings │
│  Historical ─────┘         │                        │           │
│                            ▼                        ▼           │
│                    Similarity Matrix ──▶ Ranking Algorithm      │
│                            │                        │           │
│                            ▼                        ▼           │
│                    OR-Tools Optimizer ──▶ recommendations.csv   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SERVING LAYER (FastAPI)                      │
│                                                                 │
│  ┌──────────────────┐         ┌─────────────────┐              │
│  │ CSV Artifacts    │────────▶│ Recommender     │              │
│  │ (Fast Loading)   │         │ Service         │              │
│  └──────────────────┘         └─────────────────┘              │
│           │                            │                        │
│           │                            ▼                        │
│           │                    ┌──────────────┐                 │
│           │                    │  REST API    │                 │
│           │                    │  Endpoints   │                 │
│           │                    └──────────────┘                 │
│           │                            │                        │
│           ▼                            ▼                        │
│  ┌──────────────────┐         ┌─────────────────┐              │
│  │ SQLite Database  │◀────────│  CRUD Layer     │              │
│  │ (Persistence)    │         │  (SQLAlchemy)   │              │
│  └──────────────────┘         └─────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            ┌──────────────────┐
                            │  Client Apps     │
                            │  (Web/Mobile)    │
                            └──────────────────┘
```

## ✨ Features

### Core Functionality
- **Personalized Recommendations**: Top-K internship suggestions based on student profile, skills, and preferences
- **Semantic Matching**: SBERT embeddings capture deep semantic similarity between student profiles and job descriptions
- **Hybrid Approach**: Combines content-based filtering with collaborative signals and optimization constraints
- **Rich Metadata**: Returns complete internship details including skills, stipend, location, duration, and descriptions

### API Capabilities
- ✅ Health checks and service status
- ✅ Student and internship catalog browsing
- ✅ Real-time recommendation retrieval
- ✅ Recommendation persistence to database
- ✅ Bulk data population from CSV artifacts
- ✅ Detailed internship lookup by ID

### Performance & Monitoring
- 📊 Comprehensive evaluation suite measuring:
  - **Coverage**: Percentage of students receiving recommendations
  - **Diversity**: Variety in recommendation sets (unique internships)
  - **Popularity Bias**: Gini coefficient for fairness analysis
  - **Allocation Metrics**: Utilization rates and capacity constraints
  - **API Latency**: Response time benchmarks

## 🛠️ Tech Stack

| Component | Technologies |
|-----------|-------------|
| **Backend Framework** | FastAPI, Uvicorn (ASGI) |
| **ML & Embeddings** | sentence-transformers (SBERT), scikit-learn |
| **Optimization** | OR-Tools (Google) |
| **Database** | SQLite, SQLAlchemy 2.0 (ORM) |
| **Data Processing** | Pandas, NumPy |
| **Validation** | Pydantic (schemas & validation) |
| **Testing** | Custom test suite with performance benchmarks |
| **Serialization** | Joblib (model artifacts) |

### Model Details
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
- **Vector Dimension**: 768
- **Similarity Metric**: Cosine similarity
- **Ranking Algorithm**: Score-based with optimization constraints

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- 4GB RAM minimum (for SBERT model loading)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/internship-recommender-engine.git
cd internship-recommender-engine

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Recommendations (Offline Pipeline)

Before running the API, you need to generate recommendations using the offline pipeline:

```bash
# Run the recommendation pipeline notebook
# (Ensure Jupyter is installed: pip install jupyter)
jupyter notebook notebooks/recommendation_pipeline.ipynb

# Or if using JupyterLab:
jupyter lab notebooks/recommendation_pipeline.ipynb
```

**Pipeline Steps:**
1. Load student and internship data
2. Generate SBERT embeddings for profiles and job descriptions
3. Compute similarity matrix
4. Apply ranking algorithm and optimization constraints
5. Export `recommendations.csv` to `outputs_recommender_v2/`

### Run the API Server

```bash
# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or with custom settings
uvicorn main:app --host 127.0.0.1 --port 8080 --workers 4
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Populate Database (Optional)

```bash
# Using curl
curl -X POST "http://localhost:8000/populate_db"

# Using httpie
http POST http://localhost:8000/populate_db

# Using Python requests
python -c "import requests; print(requests.post('http://localhost:8000/populate_db').json())"
```

### Test the API

```bash
# Health check
curl http://localhost:8000/

# Get recommendations for a student
curl "http://localhost:8000/recommend/S00001?top_k=5"

# Get all students
curl http://localhost:8000/students

# Get specific internship details
curl http://localhost:8000/internship/I00123
```

Access interactive API docs at: `http://localhost:8000/docs`

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Internship Recommendation API is running",
  "version": "2.0"
}
```

---

#### 2. Populate Database
```http
POST /populate_db
```

Loads CSV artifacts into SQLite database.

**Response:**
```json
{
  "message": "Database populated successfully",
  "students_loaded": 1000,
  "internships_loaded": 500,
  "recommendations_loaded": 5000
}
```

---

#### 3. Get All Students
```http
GET /students
```

**Response:**
```json
{
  "students": ["S00001", "S00002", "S00003", ...],
  "count": 1000
}
```

---

#### 4. Get All Internships
```http
GET /internships
```

**Response:**
```json
{
  "internships": [
    {
      "internship_id": "I00001",
      "title": "Software Development Intern",
      "company": "Tech Corp",
      "location": "San Francisco, CA",
      "duration_weeks": 12,
      "stipend": 5000,
      "skills_required": "Python, React, SQL",
      "description": "Work on cutting-edge web applications..."
    }
  ],
  "count": 500
}
```

---

#### 5. Get Internship Details
```http
GET /internship/{internship_id}
```

**Path Parameters:**
- `internship_id` (string): Internship identifier (e.g., "I00001")

**Response:**
```json
{
  "internship_id": "I00001",
  "title": "Machine Learning Intern",
  "company": "AI Innovations",
  "location": "Remote",
  "duration_weeks": 10,
  "stipend": 6000,
  "skills_required": "Python, TensorFlow, PyTorch",
  "description": "Build and deploy ML models...",
  "job_text": "Full job description with requirements..."
}
```

---

#### 6. Get Recommendations
```http
GET /recommend/{student_id}?top_k=10
```

**Path Parameters:**
- `student_id` (string): Student identifier (e.g., "S00001")

**Query Parameters:**
- `top_k` (integer, optional): Number of recommendations (default: 10)

**Response:**
```json
{
  "student_id": "S00001",
  "recommendations": [
    {
      "rank": 1,
      "score": 0.89,
      "internship_id": "I00234",
      "title": "Data Science Intern",
      "company": "Analytics Pro",
      "location": "Boston, MA",
      "duration_weeks": 12,
      "stipend": 5500,
      "skills_required": "Python, Pandas, SQL, Tableau",
      "description": "Analyze large datasets and build dashboards...",
      "job_text": "Complete job description..."
    },
    ...
  ],
  "count": 10
}
```

---

#### 7. Recommend and Store
```http
POST /recommend_and_store/{student_id}?top_k=10
```

Gets recommendations and persists them to the database.

**Response:**
```json
{
  "message": "Recommendations stored successfully",
  "student_id": "S00001",
  "recommendations_saved": 10
}
```

## 🔄 Recommendation Pipeline

### Offline Pipeline (Notebook-based)

The recommendation generation happens offline for optimal performance:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Data Loading                                            │
│     → students_synthetic.csv (profiles, skills, preferences)│
│     → internships_synthetic.csv (jobs, requirements, details)│
│     → historical_allocations.csv (past matches - optional)  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Feature Engineering                                     │
│     → Combine student skills, interests, academic info      │
│     → Aggregate internship requirements, descriptions       │
│     → Create rich text representations                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Embedding Generation (SBERT)                            │
│     → Encode student profiles → 768-dim vectors             │
│     → Encode internship descriptions → 768-dim vectors      │
│     → Store embeddings for similarity computation           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Similarity Computation                                  │
│     → Cosine similarity: students × internships matrix      │
│     → Shape: (num_students, num_internships)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  5. Ranking & Optimization                                  │
│     → Apply business constraints (capacity, diversity)      │
│     → OR-Tools optimization for fair allocation             │
│     → Generate top-K recommendations per student            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  6. Export Artifacts                                        │
│     → recommendations.csv (student_id, internship_id, rank, score)│
│     → evaluation_metrics.csv (quality metrics)              │
│     → Save to outputs_recommender_v2/                       │
└─────────────────────────────────────────────────────────────┘
```

### Online Serving (FastAPI)

The API serves precomputed recommendations with enrichment:

1. **Load CSV artifacts** at startup (fast in-memory access)
2. **Client requests** recommendation for student ID
3. **Filter & Sort**: Get top-K matches from `recommendations.csv`
4. **Enrich**: Join with `internships_synthetic.csv` for full details
5. **Transform**: Convert to JSON-friendly format (handle NaN, types)
6. **Return**: Pydantic-validated response with complete metadata

**Latency**: Typically < 50ms for top-10 recommendations

## 📊 Performance Evaluation

### Running the Evaluation Suite

```bash
# Run comprehensive performance analysis
python test_performance.py
```

**Output Files:**
- `performance_results/performance_report_YYYYMMDD_HHMMSS.json`
- `performance_results/performance_report_YYYYMMDD_HHMMSS.md`

### Metrics Computed

| Metric | Description | Target |
|--------|-------------|--------|
| **Coverage** | % of students receiving recommendations | > 95% |
| **Diversity** | Unique internships across all recommendations | > 80% |
| **Popularity Bias (Gini)** | Fairness of internship distribution | < 0.6 |
| **Allocation Utilization** | % of internship capacity filled | 70-90% |
| **API Latency (p95)** | 95th percentile response time | < 100ms |
| **API Latency (p99)** | 99th percentile response time | < 200ms |

### Sample Performance Report

```markdown
# Recommendation System Performance Report

**Generated:** 2025-01-20 15:30:45

## Quality Metrics
- Coverage: 98.5%
- Diversity Score: 0.87
- Popularity Bias (Gini): 0.42
- Average Score: 0.76

## Allocation Metrics
- Total Capacity: 5000 slots
- Utilized: 4250 slots (85%)
- Allocation Rate: 4.25 recommendations/student

## API Performance
- Mean Latency: 45ms
- P95 Latency: 78ms
- P99 Latency: 125ms
- Requests Tested: 100
```

## 📁 Project Structure

```
internship-recommender-engine/
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── recommendations.db               # SQLite database (generated)
│
├── app/                             # Application modules
│   ├── __init__.py
│   ├── recommender_service.py       # Core recommendation logic
│   ├── db.py                        # Database configuration
│   ├── models.py                    # SQLAlchemy ORM models
│   ├── crud.py                      # Database operations
│   └── schemas.py                   # Pydantic schemas
│
├── outputs_recommender_v2/          # Generated artifacts
│   ├── recommendations.csv          # Precomputed recommendations
│   ├── students_synthetic.csv       # Student profiles
│   ├── internships_synthetic.csv    # Internship catalog
│   ├── evaluation_metrics.csv       # Quality metrics
│   └── sbert_all_mpnet_model/       # SBERT model files
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
│
├── notebooks/                       # Jupyter notebooks
│   └── recommendation_pipeline.ipynb # Offline pipeline
│
├── tests/                           # Test suite
│   ├── test_api_response.py         # API integration tests
│   ├── test_recommendations_direct.py # Direct function tests
│   └── test_performance.py          # Performance evaluation
│
├── performance_results/             # Evaluation outputs
│   ├── performance_report_*.json
│   └── performance_report_*.md
│
└── docs/                            # Additional documentation
    ├── API.md
    ├── ARCHITECTURE.md
    └── DEPLOYMENT.md
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
SQLITE_FILE=recommendations.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_PATH=outputs_recommender_v2/sbert_all_mpnet_model
TOP_K_DEFAULT=10

# Logging
LOG_LEVEL=INFO
```

### CSV Artifact Locations

The system expects artifacts in either:
1. `outputs_recommender_v2/` (project root)
2. `outputs_recommender_v2/` (as subdirectory)

**Required Files:**
- `recommendations.csv` (student_id, internship_id, rank, score)
- `students_synthetic.csv` (student profiles)
- `internships_synthetic.csv` (internship catalog)

**Optional Files:**
- `evaluation_metrics.csv` (precomputed metrics)
- `allocations.csv` (allocation results)

## 🧪 Testing

### Manual API Testing

```bash
# Test recommendation endpoint
python tests/test_api_response.py

# Test direct recommender service
python tests/test_recommendations_direct.py

# Run performance suite
python tests/test_performance.py
```

### Interactive Testing (Swagger UI)

Navigate to `http://localhost:8000/docs` for interactive API documentation.

### Sample Test Script

```python
import requests

BASE_URL = "http://localhost:8000"

# Test health check
response = requests.get(f"{BASE_URL}/")
print("Health:", response.json())

# Get recommendations
student_id = "S00001"
response = requests.get(f"{BASE_URL}/recommend/{student_id}?top_k=5")
recommendations = response.json()

print(f"\nTop 5 Recommendations for {student_id}:")
for rec in recommendations["recommendations"]:
    print(f"{rec['rank']}. {rec['title']} at {rec['company']} (Score: {rec['score']:.2f})")
```

## 🐳 Deployment

### Docker Deployment (Recommended)

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run:**
```bash
# Build image
docker build -t internship-recommender:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/outputs_recommender_v2:/app/outputs_recommender_v2 \
  -v $(pwd)/recommendations.db:/app/recommendations.db \
  --name recommender-api \
  internship-recommender:latest

# View logs
docker logs -f recommender-api
```

### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs_recommender_v2:/app/outputs_recommender_v2
      - ./recommendations.db:/app/recommendations.db
    environment:
      - SQLITE_FILE=recommendations.db
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

**Commands:**
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

### Cloud Deployment

#### Render.com

Create `render.yaml`:
```yaml
services:
  - type: web
    name: internship-recommender-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

#### AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.10 internship-recommender

# Create environment
eb create production

# Deploy
eb deploy
```

## 🔐 Security Considerations

### Current Implementation
- ⚠️ **CORS**: Fully open (for development)
- ⚠️ **Authentication**: None
- ⚠️ **Rate Limiting**: Not implemented

### Production Recommendations

1. **Restrict CORS:**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

2. **Add API Key Authentication:**
```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.get("/recommend/{student_id}")
async def get_recommendations(
    student_id: str,
    api_key: str = Depends(api_key_header)
):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of logic
```

3. **Implement Rate Limiting:**
```bash
pip install slowapi

# In main.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/recommend/{student_id}")
@limiter.limit("100/minute")
async def get_recommendations(request: Request, student_id: str):
    # ... logic
```

## 🗺️ Roadmap

### Short-term (1-2 months)
- [ ] Add unit tests (pytest)
- [ ] Implement API authentication
- [ ] Add request/response logging
- [ ] Create comprehensive README examples
- [ ] Add health check for CSV artifacts and database

### Medium-term (3-6 months)
- [ ] Real-time embedding generation endpoint
- [ ] User feedback collection and model retraining
- [ ] A/B testing framework for recommendation algorithms
- [ ] Redis caching layer for frequently accessed data
- [ ] Prometheus metrics exporter
- [ ] Add pagination for large result sets

### Long-term (6+ months)
- [ ] Multi-model ensemble (SBERT + collaborative filtering + deep learning)
- [ ] Online learning capabilities
- [ ] Explainability features (why these recommendations?)
- [ ] Admin dashboard for monitoring
- [ ] GraphQL API option
- [ ] Multi-tenancy support for different institutions



### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 app/ tests/

# Run type checking
mypy app/

# Format code
black app/ tests/
```


### FastAPI Swagger Docs (API Endpoints Screenshot)

Auto-generated FastAPI documentation showcasing complete REST endpoints for health checks, data loading, catalog browsing, and recommendation retrieval.


<img width="1919" height="1037" alt="Screenshot 2025-11-20 223523" src="https://github.com/user-attachments/assets/ace4b47c-9a37-49b5-b84d-7f79da334e78" />







### Render Metrics Dashboard (Deploy & Monitoring Screenshot)

Real-time API performance monitoring on Render showing outbound bandwidth and system stability for the deployed Hybrid Recommender API.


<img width="1915" height="984" alt="Screenshot 2025-11-20 223418" src="https://github.com/user-attachments/assets/34a7b28a-df69-4389-9df9-27135a366489" />






### Internship Recommendation Web UI (Frontend Screenshot)

Interactive frontend displaying personalized internship recommendations powered by the Hybrid AI Recommendation Engine.


<img width="1905" height="935" alt="Screenshot 2025-11-20 223459" src="https://github.com/user-attachments/assets/689cac13-e885-40ed-974f-88aee4c47fe8" />




# Access Here :   https://intern-match-ai-gold.vercel.app/


