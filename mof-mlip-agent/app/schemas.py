"""
Pydantic data models shared across the MOF MLIP Agent.

These schemas define the structured inputs and outputs for each reasoning
stage (intent parsing, canonicalisation, novelty checking, and spec
generation). Keeping them in one place makes it easier to evolve the
agent while staying type-safe.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class QueryIntent(BaseModel):
    mof_name: Optional[str] = Field(default=None, description="MOF name if mentioned (e.g., UiO-66).")
    goal: str = Field(..., description="The core hypothesis or objective stated by the user.")
    task_hint: Optional[Literal["relaxation", "singlepoint", "adsorption_energy", "defect_energy"]] = Field(
        default=None,
        description="Best guess of MLIP-verifiable task type."
    )
    required_inputs: List[str] = Field(
        default_factory=list,
        description="Missing information needed to run an MLIP experiment (e.g., CIF path, adsorbate)."
    )
    ambiguity_flags: List[str] = Field(
        default_factory=list,
        description="Ambiguities detected (e.g., unclear structure source, unclear conditions)."
    )
    feasibility: Literal["feasible", "needs_clarification", "not_mlip_verifiable"] = Field(
        ...,
        description="Whether the query can be verified with MLIP as-is."
    )

class CanonicalQuery(BaseModel):
    query_canonical: str = Field(..., description="Rewritten query that is MLIP-verifiable and testable.")
    clarifying_questions: List[str] = Field(default_factory=list, description="Questions to ask user if needed.")

class PaperRef(BaseModel):
    title: str
    id: str
    why_relevant: str

class NoveltyVerdict(BaseModel):
    status: Literal["pass", "reject", "uncertain"]
    rationale: str
    top_refs: List[PaperRef] = Field(default_factory=list)

class ExperimentSpec(BaseModel):
    exp_id: str
    query_original: str
    query_canonical: str

    structure: Dict[str, Any]
    calculator: Dict[str, Any]
    task: Dict[str, Any]
    postprocess: Dict[str, Any]

    novelty_check: Dict[str, Any]
    notes: str
