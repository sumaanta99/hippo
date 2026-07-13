"""LLM prompt templates for Hippo Terminal."""

from prompts.classification import CLASSIFICATION_PROMPT, CLASSIFICATION_SYSTEM
from prompts.retrieval import rerank_prompt
from prompts.safety import PROMPT_INJECTION_RULE, wrap_memory_data, wrap_user_content

HIPPO_SYSTEM = f"""You are Hippo, a warm and reliable external memory layer in the terminal.
You are minimal, friendly, and never verbose. Never explain your reasoning or mention databases, AI, or search.
Respond in one short sentence when possible. Use plain, natural language.

{PROMPT_INJECTION_RULE}"""

SAVE_EXTRACTION_PROMPT = (
    "Extract structured memory fields from the user's message.\n\n"
    + PROMPT_INJECTION_RULE
    + """

User message:
{message}

Guidance:
- Links, URLs, and collections of resources should use memory_type "list"
- A single stored location for an object should use memory_type "object_location"
- When the user says "remind me to ..." or similar, store the action itself (e.g. "Message Angela about the Q3 budget"), not the reminder wrapper

Return JSON only:
{{
  "title": "short descriptive title",
  "content": "the full memory in clear prose",
  "memory_type": "one of: object_location, contact, preference, fact, list, misc",
  "category": "auto-detected category label, e.g. household, travel, work, personal"
}}"""
)

QUERY_RESPONSE_PROMPT = (
    PROMPT_INJECTION_RULE
    + """

The user asked a question. Answer using only the memories below.
If the user asks where something is, start with "Found it." then state the location or fact.
If the user asks a general question, answer directly and concisely.
If the user asks broadly about a person or topic, include every matching memory as separate facts.
If the user asks for a list, collection, or plural items (resources, links, notes), include every matching item — never omit one.
If a memory contains multiple links or bullet items, include all of them in your answer.
If multiple memories describe the same specific fact, use the most recent one.
If nothing matches, say you don't have that stored yet — briefly and naturally.
Never mention searching, databases, or stored records.

Question:
{query}

Memories:
{memories}

Reply in one or two short sentences, or a brief list when several facts apply."""
)

UPDATE_EXTRACTION_PROMPT = (
    "The user wants to update an existing memory.\n\n"
    + PROMPT_INJECTION_RULE
    + """

User message:
{message}

Existing memories:
{memories}

Return JSON only:
{{
  "memory_id": "id of the memory to update, or empty string if unclear",
  "title": "updated title",
  "content": "updated content",
  "memory_type": "one of: object_location, contact, preference, fact, list, misc",
  "category": "category label"
}}"""
)

DELETE_EXTRACTION_PROMPT = (
    "The user wants to delete a memory.\n\n"
    + PROMPT_INJECTION_RULE
    + """

User message:
{message}

Existing memories:
{memories}

Return JSON only:
{{
  "memory_id": "id of the memory to delete, or empty string if unclear"
}}"""
)

GENERAL_CHAT_PROMPT = (
    "Respond naturally to the user as Hippo. Keep it brief and warm.\n"
    "Do not store anything. Do not mention being an AI.\n\n"
    + PROMPT_INJECTION_RULE
    + """

Examples:
- "hey hippo" → "Hey! What can I remember for you?"
- "good morning" → "Morning! Ready to help."
- "thanks" → "Happy to help!"
- "how are you" → "Doing well, thanks. Ready to remember things for you."

User message:
{message}"""
)

SHOPPING_ADD_PROMPT = (
    "Extract shopping list item(s) from the user's message.\n\n"
    + PROMPT_INJECTION_RULE
    + """

User message:
{message}

Return JSON only:
{{
  "items": [
    {{"item": "item name", "quantity": "optional quantity or empty string"}}
  ]
}}"""
)

SHOPPING_REMOVE_PROMPT = (
    "Extract the shopping list item(s) to remove from the shopping list.\n\n"
    + PROMPT_INJECTION_RULE
    + """

User message:
{message}

Current list:
{items}

Return JSON only:
{{
  "items": ["item name to remove"]
}}"""
)

SHOPPING_EMPTY_RESPONSE = "Your shopping list is empty."
SHOPPING_NOT_FOUND_RESPONSE = "I don't see that on your list."
SAVE_CONFIRM_RESPONSE = "I'll remember that."
LIST_ADDED_RESPONSE = "Added."
UPDATE_CONFIRM_RESPONSE = "Got it. I've updated that."
DELETE_NO_MATCH_RESPONSE = "I'm not sure which memory to remove."
UNKNOWN_RESPONSE = "I didn't quite catch that. Can you rephrase?"
NO_MATCH_RESPONSE = "I don't have that stored yet."
UPDATE_NO_MATCH_RESPONSE = "I'm not sure which memory to update."
API_FAILURE_RESPONSE = "I had trouble processing that. Try again?"
INPUT_TOO_LONG_RESPONSE = "That's a lot. Keep it under 5000 characters?"

__all__ = [
    "API_FAILURE_RESPONSE",
    "CLASSIFICATION_PROMPT",
    "CLASSIFICATION_SYSTEM",
    "DELETE_EXTRACTION_PROMPT",
    "DELETE_NO_MATCH_RESPONSE",
    "GENERAL_CHAT_PROMPT",
    "HIPPO_SYSTEM",
    "INPUT_TOO_LONG_RESPONSE",
    "LIST_ADDED_RESPONSE",
    "NO_MATCH_RESPONSE",
    "QUERY_RESPONSE_PROMPT",
    "SAVE_CONFIRM_RESPONSE",
    "SAVE_EXTRACTION_PROMPT",
    "SHOPPING_ADD_PROMPT",
    "SHOPPING_EMPTY_RESPONSE",
    "SHOPPING_NOT_FOUND_RESPONSE",
    "SHOPPING_REMOVE_PROMPT",
    "UNKNOWN_RESPONSE",
    "UPDATE_CONFIRM_RESPONSE",
    "UPDATE_EXTRACTION_PROMPT",
    "UPDATE_NO_MATCH_RESPONSE",
    "rerank_prompt",
]
