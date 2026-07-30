"""
FastAPI service for resume screening.

Endpoints:
  POST /resumes/upload   -> parse + embed + index a resume file
  POST /rank             -> rank all indexed resumes against a job description
  GET  /resumes          -> list indexed resume ids
  DELETE /resumes/reset  -> clear the index (useful for local testing)

Run with:
  uvicorn app.main:app --reload --port 8000
"""

import os
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.parser import parse_resume_bytes
from app.embeddings import EmbeddingModel
from app.vector_store import ResumeVectorStore
from app.ranker import rank_candidates

app = FastAPI(title="Resume Screening AI", version="1.0.0")

# Allow the static frontend (served from this same app, or opened directly
# as a file, or hosted elsewhere during dev) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

embedder = EmbeddingModel.get_instance()
store = ResumeVectorStore(dimension=embedder.dimension)


class RankRequest(BaseModel):
    job_description: str = Field(..., description="Full job description text")
    required_skills: Optional[List[str]] = Field(
        default=None, description="List of must-have skills, e.g. ['Python', 'AWS']"
    )
    min_years_required: Optional[float] = Field(
        default=None, description="Minimum years of experience the JD asks for"
    )
    top_k: int = Field(default=10, ge=1, le=100)


class RankedCandidate(BaseModel):
    resume_id: str
    final_score: float
    semantic_score: float
    skill_overlap_score: float
    experience_match_score: float
    matched_skills: List[str]
    missing_skills: List[str]


class RankResponse(BaseModel):
    job_description_preview: str
    total_candidates_considered: int
    rankings: List[RankedCandidate]


@app.post("/resumes/upload")
async def upload_resume(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        parsed = parse_resume_bytes(contents, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not parsed.raw_text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in file.")

    embedding = embedder.encode(parsed.raw_text)[0]
    metadata = {
        "resume_id": parsed.resume_id,
        "raw_text": parsed.raw_text,
        "sections": parsed.sections,
        "filename": file.filename,
    }
    idx = store.add(embedding, metadata)
    store.save()

    return {
        "status": "indexed",
        "resume_id": parsed.resume_id,
        "vector_index": idx,
        "total_indexed": len(store),
    }


@app.post("/rank", response_model=RankResponse)
async def rank_resumes(request: RankRequest):
    if len(store) == 0:
        raise HTTPException(status_code=404, detail="No resumes indexed yet. Upload some first.")

    jd_embedding = embedder.encode(request.job_description)[0]
    search_results = store.search(jd_embedding, top_k=request.top_k)

    ranked = rank_candidates(
        search_results,
        required_skills=request.required_skills,
        min_years_required=request.min_years_required,
    )

    return RankResponse(
        job_description_preview=request.job_description[:200],
        total_candidates_considered=len(search_results),
        rankings=[
            RankedCandidate(
                resume_id=r.resume_id,
                final_score=r.final_score,
                semantic_score=r.semantic_score,
                skill_overlap_score=r.skill_overlap_score,
                experience_match_score=r.experience_match_score,
                matched_skills=r.matched_skills,
                missing_skills=r.missing_skills,
            )
            for r in ranked
        ],
    )


@app.get("/resumes")
async def list_resumes():
    return {
        "total": len(store),
        "resume_ids": [m["resume_id"] for m in store.metadata],
    }


@app.delete("/resumes/reset")
async def reset_index():
    global store
    store = ResumeVectorStore(dimension=embedder.dimension,
                               index_path=store.index_path,
                               meta_path=store.meta_path)
    store.index.reset()
    store.metadata = []
    store.save()
    return {"status": "reset"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": embedder.model_name, "indexed_resumes": len(store)}


# Serve the frontend UI at http://localhost:8000/  (mounted last so it
# doesn't shadow the API routes above)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")