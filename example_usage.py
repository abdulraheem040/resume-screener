"""
Standalone example that exercises the full pipeline WITHOUT needing the
FastAPI server running -- useful for quick local testing or a notebook.

Usage:
    python example_usage.py
"""

from app.parser import parse_resume_file, ParsedResume
from app.embeddings import EmbeddingModel
from app.vector_store import ResumeVectorStore
from app.ranker import rank_candidates


def build_fake_resume(resume_id: str, text: str) -> ParsedResume:
    from app.parser import clean_text, split_sections
    cleaned = clean_text(text)
    return ParsedResume(resume_id=resume_id, raw_text=cleaned, sections=split_sections(cleaned))


def main():
    embedder = EmbeddingModel.get_instance()
    store = ResumeVectorStore(
        dimension=embedder.dimension,
        index_path="/tmp/demo_resume_index.faiss",
        meta_path="/tmp/demo_resume_meta.pkl",
    )

    sample_resumes = {
        "alice.txt": """
            Summary
            Backend engineer with 5 years experience building scalable APIs.

            Skills
            Python, FastAPI, PostgreSQL, Docker, AWS, Kubernetes

            Experience
            Senior Backend Engineer at Acme Corp, 2021-present.
            Built microservices handling 10M requests/day.
        """,
        "bob.txt": """
            Summary
            Frontend developer, 2 years experience, React specialist.

            Skills
            JavaScript, React, CSS, HTML, Figma

            Experience
            Frontend Developer at Widgets Inc, 2023-present.
        """,
        "carol.txt": """
            Summary
            Full-stack engineer, 7 years experience.

            Skills
            Python, Django, React, AWS, PostgreSQL, Docker

            Experience
            Lead Engineer at DataCo, 2018-present.
            Led migration to AWS, managed team of 4 engineers.
        """,
    }

    for resume_id, text in sample_resumes.items():
        parsed = build_fake_resume(resume_id, text)
        embedding = embedder.encode(parsed.raw_text)[0]
        store.add(embedding, {
            "resume_id": parsed.resume_id,
            "raw_text": parsed.raw_text,
            "sections": parsed.sections,
        })

    job_description = """
        We are hiring a Senior Backend Engineer with strong Python experience.
        Requirements: Python, AWS, Docker, PostgreSQL. 5+ years of experience
        building production APIs at scale.
    """

    jd_embedding = embedder.encode(job_description)[0]
    results = store.search(jd_embedding, top_k=10)

    ranked = rank_candidates(
        results,
        required_skills=["Python", "AWS", "Docker", "PostgreSQL"],
        min_years_required=5,
    )

    print(f"\nRanking candidates for job description:\n{job_description.strip()}\n")
    print("-" * 70)
    for r in ranked:
        print(f"{r.resume_id:15s} | final={r.final_score:.3f} "
              f"semantic={r.semantic_score:.3f} "
              f"skills={r.skill_overlap_score:.3f} "
              f"exp={r.experience_match_score:.3f}")
        print(f"   matched: {r.matched_skills} | missing: {r.missing_skills}")
    print("-" * 70)


if __name__ == "__main__":
    main()
