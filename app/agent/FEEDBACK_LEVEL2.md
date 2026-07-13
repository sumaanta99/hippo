"""Level 2 feedback application — design notes (not implemented).

Periodic job clusters `not_helpful` feedback by tool-call pattern, e.g.
"update_memory without prior search_memory". Output is a human-readable
suggestion to edit the system prompt or tool descriptions — never auto-applied.

Suggested implementation:
1. Nightly SQLite query groups feedback.tool_calls_json by normalized sequence.
2. Flag groups above a count threshold with representative user_message + note.
3. Emit report to admin analytics or operator email/webhook.
4. Operator reviews and manually updates `agent/prompts.py` or `agent/tools.py`.

Keep humans in the loop: noisy feedback should not silently rewrite production prompts.
"""
