"""
Chain for parsing the user's raw query into a structured QueryIntent.

This is the first reasoning step in the agent: it decides what the user
is actually asking for and whether it is MLIP-verifiable as stated.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.schemas import QueryIntent


SYSTEM = """You are an expert scientific agent for MOF(Metal Organic Framework) simulations.
Your job is to judge whether the user's query is verifiable using ML interatomic potentials (MLIP).
MLIP(Machine Learning Inter-atomic Potential)-verifiable tasks include: geometry relaxation, single-point energy/forces/stress,
and well-defined comparative energies (e.g., adsorption energy) if the structure/species are specified.

Not MLIP-verifiable as-is: vague claims about synthesize-ability, real-world adsorption isotherms without a defined protocol,
or unspecified structures/conditions.

You also receive PAST_RUN memory from previous executions. Use it to stay consistent and avoid repeated work.
All output must be English only.
Return a strict structured output.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", """User query:
{query}

PAST_RUN memory (may be empty):
{memory_context}

Extract:
- MOF name if any
- The core goal
- Best task_hint
- required_inputs (missing)
- ambiguity_flags
- feasibility (feasible / needs_clarification / not_mlip_verifiable)
""")
])


def build_intent_chain(llm: ChatOpenAI):
    """
    Create a Runnable chain that takes:
        {"query": str, "memory_context": str}
    and returns a structured QueryIntent.
    """
    return PROMPT | llm.with_structured_output(QueryIntent)


def intent_to_jsonable(intent: QueryIntent) -> dict:
    """
    Convenience helper to turn a QueryIntent into a plain JSON-serialisable dict.
    """
    return intent.model_dump()
