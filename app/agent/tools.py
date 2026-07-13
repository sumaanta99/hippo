"""Claude tool schemas for the Hippo agent loop."""

from __future__ import annotations

from typing import Any

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "save_memory",
        "description": (
            "Store a new memory from the user's message. Use for locations, "
            "follow-ups, deadlines, contacts, facts, and list items. Store the "
            "action itself without reminder wrappers like 'Remind me to'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "What to remember, in concise natural language.",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "object_location",
                        "follow_up",
                        "deadline",
                        "contact",
                        "fact",
                        "list_item",
                    ],
                    "description": "Optional memory category.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_memory",
        "description": (
            "Search stored memories by meaning or keywords. Always call this "
            "before update_memory or delete_memory when the target memory_id "
            "is not already known from context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of matches to return.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "update_memory",
        "description": (
            "Update an existing memory by id. Call search_memory first when "
            "multiple memories could match or the id is unknown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Id of the memory to update.",
                },
                "new_content": {
                    "type": "string",
                    "description": "Replacement content for the memory.",
                },
            },
            "required": ["memory_id", "new_content"],
        },
    },
    {
        "name": "delete_memory",
        "description": (
            "Delete a memory by id. Call search_memory first when the target "
            "is ambiguous."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Id of the memory to delete.",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "add_shopping_item",
        "description": "Add one or more items to the shopping list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": "Item name to add.",
                },
            },
            "required": ["item"],
        },
    },
    {
        "name": "remove_shopping_item",
        "description": "Remove an item from the shopping list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": "Item name to remove.",
                },
            },
            "required": ["item"],
        },
    },
    {
        "name": "list_shopping",
        "description": "Show all items currently on the shopping list.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
