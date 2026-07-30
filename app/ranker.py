"""
Ranking logic.

Pure cosine similarity from embeddings is a weak signal on its own --
it rewards verbose overlap in wording rather than actual fit. This
module blends semantic similarity with:
  - explicit required-skill keyword overlap
  - years-of-experience match against a JD requirement (if extractable)

Weights are configurable so you can tune against a labeled validation
set of (resume, JD, human_fit_score) pairs.
"""

import re
from dataclasses import dataclass


DEFAULT_WEIGHTS = {
    "semantic": 0.6,
    "skill_overlap": 0.25,
    "experience_match": 0.15,
}


@dataclass
class RankedResult:
    resume_id: str
    final_score: float
    semantic_score: float
    skill_overlap_score: float
    experience_match_score: float
    matched_skills: list
    missing_skills: list


def extract_skills_from_text(text: str, skill_vocabulary: list) -> set:
    """Case-insensitive substring match against a provided skill vocabulary."""
    text_lower = text.lower()
    found = set()
    for skill in skill_vocabulary:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def skill_overlap_score(resume_text: str, required_skills: list) -> tuple:
    if not required_skills:
        return 1.0, [], []  # no requirement specified -> don't penalize
    found = extract_skills_from_text(resume_text, required_skills)
    missing = [s for s in required_skills if s not in found]
    score = len(found) / len(required_skills)
    return score, sorted(found), missing


def extract_years_experience(text: str) -> float:
    """Grabs the largest 'N years' mention as a rough proxy for total experience."""
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text.lower())
    years = [float(m) for m in matches]
    return max(years) if years else 0.0


def experience_match_score(resume_text: str, min_years_required: float) -> float:
    if not min_years_required or min_years_required <= 0:
        return 1.0
    candidate_years = extract_years_experience(resume_text)
    if candidate_years >= min_years_required:
        return 1.0
    # partial credit if close, tapering to 0
    return max(0.0, candidate_years / min_years_required)


def rank_candidates(
    search_results: list,
    required_skills: list = None,
    min_years_required: float = None,
    weights: dict = None,
) -> list:
    """
    search_results: output of ResumeVectorStore.search(), each item has
        {"score": cosine_sim, "metadata": {"resume_id", "raw_text", ...}}
    """
    weights = weights or DEFAULT_WEIGHTS
    required_skills = required_skills or []
    ranked = []

    for result in search_results:
        meta = result["metadata"]
        resume_text = meta.get("raw_text", "")
        semantic = result["score"]

        skill_score, matched, missing = skill_overlap_score(resume_text, required_skills)
        exp_score = experience_match_score(resume_text, min_years_required)

        final = (
            weights["semantic"] * semantic
            + weights["skill_overlap"] * skill_score
            + weights["experience_match"] * exp_score
        )

        ranked.append(
            RankedResult(
                resume_id=meta.get("resume_id", "unknown"),
                final_score=round(float(final), 4),
                semantic_score=round(float(semantic), 4),
                skill_overlap_score=round(float(skill_score), 4),
                experience_match_score=round(float(exp_score), 4),
                matched_skills=matched,
                missing_skills=missing,
            )
        )

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
