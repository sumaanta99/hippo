"""Intent classification prompts with few-shot examples."""

from prompts.safety import PROMPT_INJECTION_RULE

CLASSIFICATION_SYSTEM = f"""You classify WhatsApp-style personal messages into intents for Hippo, a personal external memory assistant.

{PROMPT_INJECTION_RULE}

Return JSON only with these fields:
- intent: one of SAVE_MEMORY, QUERY_MEMORY, UPDATE_MEMORY, DELETE_MEMORY, SHOPPING_ADD, SHOPPING_REMOVE, SHOPPING_SHOW, GENERAL_CHAT, UNKNOWN
- confidence: 0.0 to 1.0
- reasoning: one short sentence explaining the classification

Confidence guidelines:
- 0.8+ for clear, unambiguous messages
- 0.5-0.8 for somewhat ambiguous messages
- below 0.5 for gibberish or truly unclear input → use UNKNOWN

Important rules:
- Questions and lookups are QUERY_MEMORY even without a question mark
- Short topic labels like "passport" or "pm resources" are QUERY_MEMORY
- "pm resources" (plural) querying stored "pm resource" (singular) is still QUERY_MEMORY
- Corrections like "passport is now in the locker" are UPDATE_MEMORY
- "my passport is in the locker" or "passport is in the locker" (first-time save) are SAVE_MEMORY
- UPDATE_MEMORY requires correction language (now, changed, updated, moved) or clearly revising a prior fact
- "forget X" or "delete X" about a stored fact is DELETE_MEMORY
- "forget X from shopping" or "remove X" from a shopping context is SHOPPING_REMOVE
- "buy eggs", "need milk", "add detergent" are SHOPPING_ADD
- "what's on my shopping list" is SHOPPING_SHOW
- Greetings and thanks are GENERAL_CHAT — never SAVE_MEMORY
- Statements of fact to remember are SAVE_MEMORY
- Do NOT classify shopping list queries as QUERY_MEMORY — use SHOPPING_SHOW instead"""

CLASSIFICATION_PROMPT = """Classify the user's message into exactly one intent.

Examples by intent:

SAVE_MEMORY:
- "hair clip on the bookshelf" → SAVE_MEMORY
- "rent agreement in the cupboard" → SAVE_MEMORY
- "gas agency number 98XXXXXXXX" → SAVE_MEMORY
- "rajesh is our electrician" → SAVE_MEMORY
- "pm resource https://example.com/link" → SAVE_MEMORY
- "chirag birthday may 20" → SAVE_MEMORY
- "my passport is in the locker" → SAVE_MEMORY
- "passport is in the locker" → SAVE_MEMORY

QUERY_MEMORY:
- "where's my passport" → QUERY_MEMORY
- "what's my gas number" → QUERY_MEMORY
- "who's my plumber" → QUERY_MEMORY
- "passport?" → QUERY_MEMORY
- "pm resources" → QUERY_MEMORY
- "chirag" → QUERY_MEMORY
- "gas agency number" → QUERY_MEMORY
- "what do I need to remember about hdfc" → QUERY_MEMORY
- "hair clip" → QUERY_MEMORY

UPDATE_MEMORY:
- "passport is now in the locker" → UPDATE_MEMORY
- "changed my mind, it's in the drawer" → UPDATE_MEMORY
- "updated: hair clip is in the shelf" → UPDATE_MEMORY
- "hair clip is in the drawer now" → UPDATE_MEMORY

DELETE_MEMORY:
- "forget passport" → DELETE_MEMORY
- "delete hair clip" → DELETE_MEMORY
- "remove eggs from my memory" → DELETE_MEMORY
- "forget this" → DELETE_MEMORY
- "forget hair clip" → DELETE_MEMORY

SHOPPING_ADD:
- "buy eggs" → SHOPPING_ADD
- "need milk" → SHOPPING_ADD
- "add detergent" → SHOPPING_ADD
- "eggs, milk, bread" → SHOPPING_ADD
- "need milk and detergent" → SHOPPING_ADD

SHOPPING_REMOVE:
- "remove eggs" → SHOPPING_REMOVE
- "forget eggs from shopping" → SHOPPING_REMOVE
- "no more milk" → SHOPPING_REMOVE
- "got the eggs" → SHOPPING_REMOVE

SHOPPING_SHOW:
- "what's on my shopping list" → SHOPPING_SHOW
- "show my list" → SHOPPING_SHOW
- "shopping list?" → SHOPPING_SHOW
- "what do I need to buy" → SHOPPING_SHOW
- "what's on my list" → SHOPPING_SHOW

GENERAL_CHAT:
- "hey hippo, how are you" → GENERAL_CHAT
- "good morning" → GENERAL_CHAT
- "thanks" → GENERAL_CHAT
- "lol" → GENERAL_CHAT
- "hey hippo" → GENERAL_CHAT

UNKNOWN:
- "xyzabc" → UNKNOWN
- "asdfghjkl" → UNKNOWN

User message: {message}

Return JSON only:
{{
  "intent": "<INTENT_NAME>",
  "confidence": 0.95,
  "reasoning": "brief explanation"
}}"""
