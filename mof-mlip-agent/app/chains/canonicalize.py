"""
Chain for turning a free-form query + parsed intent into a CanonicalQuery.

The canonical representation is a single, precise sentence that is easier
to use for retrieval and experiment specification than the raw user text.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.schemas import CanonicalQuery, QueryIntent


SYSTEM = """You are a MOF MLIP query canonicalizer.
You receive:
- The user's original free-form query about a MOF MLIP experiment.
- A parsed intent object (JSON) describing what the user wants.
- PAST_RUN memory providing prior canonical queries and specs.

Your job:
- Rewrite the query into a precise, unambiguous canonical form suitable for:
  - literature search
  - RAG over local simulation specs
  - MLIP experiment specification
- Keep only essential details (structure, task type, key conditions).
- Remove chit-chat and redundant wording.
- If the intent is not MLIP-feasible, still produce the "closest" MLIP-feasible canonical question.

All output must be English only.
Return a strict structured CanonicalQuery object.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", """Original query:
{query}

Parsed intent (JSON):
{intent_json}

PAST_RUN memory (may be empty):
{memory_context}

Rewrite the original query into a single canonical experimental question that:
- is as specific as possible about the MOF/structure
- clearly states the MLIP task (e.g., relaxation, singlepoint, adsorption_energy, defect_energy)
- notes key numerical parameters when present (cutoffs, fmax, steps, etc.)

Return only a structured CanonicalQuery.
""")
])


def build_canonicalize_chain(llm: ChatOpenAI):
    """
    Create a Runnable chain that maps:
        {"query": str, "intent_json": dict, "memory_context": str}
    -> CanonicalQuery
    """
    return PROMPT | llm.with_structured_output(CanonicalQuery)


def intent_to_jsonable(intent: QueryIntent) -> dict:
    """
    Serialize a QueryIntent into a plain dict for injection into prompts.
    """
    return intent.model_dump()
