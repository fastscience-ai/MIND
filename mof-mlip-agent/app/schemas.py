"""
Pydantic data models shared across the MoF(Metal-organic Framework) MLIP(Machine Learning interatomic Potential) Agent.

These schemas define the structured inputs and outputs for each reasoning
stage (intent parsing, canonicalisation, novelty checking, and spec
generation). Keeping them in one place makes it easier to evolve the
agent while staying type-safe.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class QueryIntent(BaseModel):
    mof_name: Optional[str] = Field(default=None, description="MoF(Metal-organic Framework) name if mentioned (e.g., UiO-66).")
    goal: str = Field(..., description="The core hypothesis or objective stated by the user.")
    task_hint: Optional[Literal["relaxation", "singlepoint", "adsorption_energy", "defect_energy"]] = Field(
        default=None,
        description="Best guess of MLIP(Machine Learning interatomic Potential)-verifiable task type."
    )
    required_inputs: List[str] = Field(
        default_factory=list,
        description="Missing information needed to run an MLIP(Machine Learning interatomic Potential) experiment (e.g., CIF path, adsorbate)."
    )
    ambiguity_flags: List[str] = Field(
        default_factory=list,
        description="Ambiguities detected (e.g., unclear structure source, unclear conditions)."
    )
    feasibility: Literal["feasible", "needs_clarification", "not_mlip_verifiable"] = Field(
        ...,
        description="Whether the query can be verified with MLIP(Machine Learning interatomic Potential) as-is."
    )

class CanonicalQuery(BaseModel):
    query_canonical: str = Field(..., description="Rewritten query that is MLIP(Machine Learning interatomic Potential)-verifiable and testable.")
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

    # Optional with defaults so parsing succeeds when the LLM omits nested objects (e.g. with function_calling).
    structure: Dict[str, Any] = Field(default_factory=dict, description="Structure id, format, path.")
    calculator: Dict[str, Any] = Field(default_factory=dict, description="Engine, model, precision.")
    task: Dict[str, Any] = Field(default_factory=dict, description="Task type and parameters.")
    postprocess: Dict[str, Any] = Field(default_factory=dict, description="Outputs and save_trajectory.")
    novelty_check: Dict[str, Any] = Field(default_factory=dict, description="Status and top_refs.")

    notes: str = Field(..., description="Short specific notes.")
