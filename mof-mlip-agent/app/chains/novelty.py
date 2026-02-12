"""
Chain responsible for novelty checking of proposed MoF(Metal-organic Framework) MLIP(Machine Learning interatomic Potential) experiments.

Given a canonical question, memory context, and retrieved literature /
local RAG snippets, the chain returns a NoveltyVerdict describing whether
the idea should be rejected, passed, or marked uncertain.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.schemas import NoveltyVerdict


SYSTEM = """You are a novelty gate for MLIP(Machine Learning interatomic Potential)-based MoF(Metal-organic Framework) hypotheses.
Given a canonical experimental question and retrieved literature snippets,
decide:
- reject: if the same claim/experiment appears already established with high confidence
- pass: if no strong prior art is found or the proposed test differs materially
- uncertain: if evidence is ambiguous

You also receive PAST_RUN memory. If an identical or near-identical past run was already rejected/passed,
use that to maintain consistency, but still rely on the provided literature when possible.

Be conservative: only reject when it is clearly already established.
English only.
Return structured output.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", """Canonical experimental question:
{canonical}

PAST_RUN memory:
{memory_context}

Retrieved literature (may be partial):
{lit}

Local RAG context (optional):
{local_ctx}

Return novelty verdict with rationale and up to 3 top references.
Use IDs from the literature snippet when possible.
""")
])


def build_novelty_chain(llm: ChatOpenAI):
    """
    Create a Runnable chain that maps:
        {"canonical": str, "memory_context": str, "lit": str, "local_ctx": str}
    -> NoveltyVerdict
    """
    return PROMPT | llm.with_structured_output(NoveltyVerdict, method="function_calling")
