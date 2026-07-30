# Resume Screening AI

Ranks resumes against a job description using semantic search
(Sentence Transformers + FAISS), blended with explicit skill/experience
matching, served through a FastAPI API.

## Project structure

```
resume_screener/
├── app/
│   ├── parser.py          # PDF/DOCX/TXT parsing + section splitting
│   ├── embeddings.py       # Sentence-Transformers wrapper
│   ├── vector_store.py     # FAISS index + metadata management
│   ├── ranker.py           # semantic + skill + experience scoring
│   └── main.py             # FastAPI app (upload / rank / list / reset)
├── frontend/
│   └── index.html          # single-page UI, visualizes the whole pipeline
├── example_usage.py         # run the pipeline directly, no server needed
├── requirements.txt
└── README.md
```

## Frontend

`frontend/index.html` is a self-contained UI (no build step, no npm) that:

- lets you drop in resumes and type a job description, required skills, and minimum years
- shows a 4-stage pipeline strip (**Parse → Embed → Search → Rank**) that lights up live as your request moves through the backend
- renders a **match radar**: each ranked candidate is plotted as a blip whose distance from the center is their semantic similarity to the job description — closer means a better match
- shows a ranked leaderboard with a score breakdown bar per candidate (semantic / skill overlap / experience match) and matched vs. missing required skills

It's served automatically by the FastAPI app — the static mount in `app/main.py` picks it up from the `frontend/` folder, so once `uvicorn` is running you don't need anything extra.

```bash
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000/** in your browser. The status pill top-right shows whether the API is reachable and how many resumes are currently indexed.

If you'd rather open `frontend/index.html` directly as a file (double-click it) instead of through the server, it still works — CORS is enabled on the backend, and the page falls back to calling `http://localhost:8000` automatically. Just make sure `uvicorn` is running separately in that case.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run will download the `all-mpnet-base-v2` model (~420MB) from
Hugging Face — needs internet access once, then it's cached locally.

## Quick local test (no server)

```bash
python example_usage.py
```

This indexes three fake resumes and ranks them against a sample backend
job description, printing the score breakdown per candidate.

## Running the API

```bash
uvicorn app.main:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for interactive Swagger UI.

### Upload a resume

```bash
curl -X POST "http://localhost:8000/resumes/upload" \
  -F "file=@/path/to/resume.pdf"
```

### Rank resumes against a job description

```bash
curl -X POST "http://localhost:8000/rank" \
  -H "Content-Type: application/json" \
  -d '{
        "job_description": "Senior backend engineer. Python, AWS, Docker, PostgreSQL required. 5+ years experience.",
        "required_skills": ["Python", "AWS", "Docker", "PostgreSQL"],
        "min_years_required": 5,
        "top_k": 10
      }'
```

### List indexed resumes

```bash
curl http://localhost:8000/resumes
```

### Reset the index (dev/testing)

```bash
curl -X DELETE http://localhost:8000/resumes/reset
```

## How ranking works

Final score is a weighted blend (weights in `app/ranker.py`):

| Component          | Weight | What it captures                                   |
|---------------------|--------|-----------------------------------------------------|
| Semantic similarity | 0.60   | Cosine similarity between JD and resume embeddings   |
| Skill overlap        | 0.25   | Fraction of `required_skills` found in the resume    |
| Experience match      | 0.15   | Candidate's stated years vs. `min_years_required`     |

Tune these weights against a labeled set of (resume, JD, human judgment)
pairs if you have one — raw cosine similarity alone tends to reward
verbose wording overlap rather than true fit.

## Scaling notes

- **>50k resumes**: swap `IndexFlatIP` in `vector_store.py` for
  `IndexIVFFlat` or `IndexHNSWFlat` for faster approximate search.
- **Metadata**: currently stored as an in-memory pickle alongside the
  FAISS index. For production, move `raw_text`/`sections` into
  Postgres (or Postgres + `pgvector`) and keep FAISS purely for vector
  search, joining by `resume_id`.
- **Bias**: semantic similarity can pick up unwanted proxies (school
  names, employment gaps, name patterns). Audit for fairness before
  using this to make real hiring decisions — treat scores as a
  screening aid for a human, not an automated decision.

## Extending

- Add authentication before exposing `/resumes/upload` publicly.
- Add a `/rank/batch` endpoint that accepts multiple JDs at once.
- Fine-tune the Sentence Transformer on labeled (resume, JD, fit) pairs
  using `sentence-transformers`' `CosineSimilarityLoss` if you have
  historical hiring outcome data.