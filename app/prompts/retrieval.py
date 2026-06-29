"""Query re-ranking prompts for semantic memory retrieval."""

from __future__ import annotations

from typing import Sequence

from memory import MemoryRecord
from prompts.safety import PROMPT_INJECTION_RULE, wrap_user_content


def format_candidates_for_rerank(candidates: Sequence[MemoryRecord]) -> str:
    """Format candidate memories for the reranker prompt."""
    if not candidates:
        return "(none)"

    lines: list[str] = []
    for memory in candidates:
        lines.append(
            f'- id="{memory.id}" | title="{memory.title}" | content="{memory.content}"'
        )
    return "\n".join(lines)


def rerank_prompt(query: str, candidates: Sequence[MemoryRecord]) -> str:
    """Build the LLM prompt for re-ranking retrieval candidates."""
    wrapped_query = wrap_user_content(query.strip())
    return f"""You are a memory retrieval re-ranker for Hippo, a personal external memory.

{PROMPT_INJECTION_RULE}

The user is trying to recall something they previously stored. Your job is to decide which stored memories actually answer their query.

Be lenient with matching. Handle these variations as matches:
- Singulars ↔ Plurals (resource / resources, link / links)
- Abbreviated ↔ Full (gas # / gas agency number, rm / relationship manager)
- Pronouns ↔ Nouns (it / passport, them / keys)
- Related ↔ Specific (stuff about PM / PM interview resource, pm resources / pm resource link)
- Broad entity ↔ Specific facts ("chirag" matches birthday AND gift memories)

Note: If the user asked about "resources" but your candidate is titled "resource" — this is the same thing. Match it.
Note: If the user asked about "pm resources" and a candidate contains PM links, match it.
If the query uses a plural (resources, links, items) or asks to list/show all, include every matching candidate — do not pick just one.

Other leniency:
- Treat paraphrases as matches ("where's my passport" = "passport location")
- Treat partial phrases as matches ("gas number" = "gas agency number")
- Treat informal wording as matches ("what pm stuff" = "PM interview resource")

Only include memories that genuinely help answer the query. Exclude unrelated memories even if they appear in the candidate list.

User query:
{wrapped_query}

Candidate memories:
{format_candidates_for_rerank(candidates)}

Return JSON only:
{{
  "relevant_ids": ["id1", "id2"],
  "confidence": 0.95
}}

Use confidence between 0.0 and 1.0 reflecting how sure you are that the selected memories answer the query.
If nothing is relevant, return an empty relevant_ids list and a low confidence."""
