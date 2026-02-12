"""
Chain that turns a canonical query and novelty information into an
ExperimentSpec suitable for downstream execution (e.g. SevenNet).
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.schemas import ExperimentSpec, NoveltyVerdict


SYSTEM = """You generate an MLIP experiment JSON spec for MOF calculations.
You also receive PAST_RUN memory to keep formatting and assumptions consistent across runs.

Assume an MLIP engine like SevenNet. You must output a structured ExperimentSpec only.
English only.

Rules:
- If the user did not specify a structure path, put a placeholder path and record it in notes.
- task.type should be one of: relaxation, singlepoint, adsorption_energy, defect_energy.
- Keep defaults sensible: relaxation fmax=0.05, max_steps=500 if not specified.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", """Original query:
{query_original}

Canonical query:
{query_canonical}

PAST_RUN memory:
{memory_context}

Novelty:
{novelty_json}

exp_id:
{exp_id}

Generate an ExperimentSpec with:
- structure: id/format/path
- calculator: engine/model/precision
- task: type + parameters
- postprocess: outputs + save_trajectory
- novelty_check: status + top_refs
- notes: short and specific
""")
])


def build_spec_chain(llm: ChatOpenAI):
    """
    Create a Runnable chain that maps:
        {
          "query_original": str,
          "query_canonical": str,
          "memory_context": str,
          "novelty_json": dict,
          "exp_id": str,
        }
    -> ExperimentSpec
    """
    return PROMPT | llm.with_structured_output(ExperimentSpec, method="function_calling")


def novelty_to_jsonable(n: NoveltyVerdict) -> dict:
    """
    Helper to convert a NoveltyVerdict into a plain dict for prompting.
    """
    return n.model_dump()
