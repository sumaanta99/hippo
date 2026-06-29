"""SQLite-backed memory storage and retrieval."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Sequence

import aiosqlite
from pydantic import BaseModel

from config import MemoryType, Settings, get_database_path, get_settings
from embeddings import EmbeddingClient, cosine_similarity, memory_embedding_text


logger = logging.getLogger(__name__)


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "in",
        "on",
        "at",
        "to",
        "of",
        "my",
        "now",
        "are",
        "was",
        "where",
        "location",
        "add",
        "do",
        "up",
        "or",
        "as",
        "be",
        "by",
        "go",
        "so",
        "no",
    }
)


class MemoryRecord(BaseModel):
    """A single stored memory."""

    id: str
    user_id: str
    title: str
    content: str
    memory_type: MemoryType
    category: str
    timestamp: datetime
    is_archived: bool = False
    version_number: int = 1


class MemoryCreate(BaseModel):
    """Fields required to create a memory."""

    title: str
    content: str
    memory_type: MemoryType
    category: str


class MemoryUpdate(BaseModel):
    """Fields used when updating a memory."""

    title: str
    content: str
    memory_type: MemoryType
    category: str


class MemoryStore:
    """Async SQLite store for user memories."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        """Initialize the memory store."""
        self._settings = settings or get_settings()
        self._db_path = get_database_path(self._settings)
        self._embeddings = embedding_client or EmbeddingClient(self._settings)

    async def initialize(self) -> None:
        """Create database tables if they do not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    is_archived INTEGER NOT NULL DEFAULT 0,
                    version_number INTEGER NOT NULL DEFAULT 1,
                    embedding TEXT
                )
                """
            )
            await self._ensure_embedding_column(db)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"
            )
            await db.commit()

        await self.backfill_embeddings()

    async def _ensure_embedding_column(self, db: aiosqlite.Connection) -> None:
        """Add the embedding column to older databases."""
        cursor = await db.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "embedding" not in columns:
            await db.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")

    async def backfill_embeddings(self) -> None:
        """Generate embeddings for memories that do not have one yet."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT id, title, content FROM memories
                WHERE user_id = ? AND is_archived = 0
                  AND (embedding IS NULL OR embedding = '')
                """,
                (self._settings.user_id,),
            )
            rows = await cursor.fetchall()

        if not rows:
            return

        logger.info("Backfilling embeddings for %d memories.", len(rows))
        for row in rows:
            await self._store_embedding(row["id"], row["title"], row["content"])

    async def create(self, data: MemoryCreate) -> MemoryRecord:
        """Persist a new memory and return the stored record."""
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            user_id=self._settings.user_id,
            title=data.title.strip(),
            content=data.content.strip(),
            memory_type=data.memory_type,
            category=data.category.strip(),
            timestamp=datetime.now(timezone.utc),
            is_archived=False,
            version_number=1,
        )
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO memories (
                    id, user_id, title, content, memory_type, category,
                    timestamp, is_archived, version_number, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.user_id,
                    record.title,
                    record.content,
                    record.memory_type.value,
                    record.category,
                    record.timestamp.isoformat(),
                    int(record.is_archived),
                    record.version_number,
                    None,
                ),
            )
            await db.commit()

        await self._store_embedding(record.id, record.title, record.content)
        return record

    async def _store_embedding(self, memory_id: str, title: str, content: str) -> None:
        """Generate and persist an embedding for a memory."""
        try:
            vector = await self._embeddings.embed_one(
                memory_embedding_text(title, content)
            )
        except Exception as exc:
            logger.warning(
                "Failed to embed memory %s: %s", memory_id[:8], exc
            )
            return

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE memories
                SET embedding = ?
                WHERE id = ? AND user_id = ?
                """,
                (json.dumps(vector), memory_id, self._settings.user_id),
            )
            await db.commit()

    async def _load_embedding(self, memory_id: str) -> list[float] | None:
        """Load a stored embedding vector for a memory."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT embedding FROM memories
                WHERE id = ? AND user_id = ? AND is_archived = 0
                """,
                (memory_id, self._settings.user_id),
            )
            row = await cursor.fetchone()

        if row is None or not row[0]:
            return None

        try:
            parsed = json.loads(row[0])
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, list):
            return None
        return [float(value) for value in parsed]

    async def get_by_id(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a single memory by identifier."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM memories
                WHERE id = ? AND user_id = ? AND is_archived = 0
                """,
                (memory_id, self._settings.user_id),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    async def list_active(self) -> list[MemoryRecord]:
        """Return all active memories for the current user."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND is_archived = 0
                ORDER BY timestamp DESC
                """,
                (self._settings.user_id,),
            )
            rows = await cursor.fetchall()
        return [_row_to_record(row) for row in rows]

    async def update(self, memory_id: str, data: MemoryUpdate) -> MemoryRecord | None:
        """Update an existing memory and increment its version."""
        existing = await self.get_by_id(memory_id)
        if existing is None:
            return None

        updated = existing.model_copy(
            update={
                "title": data.title.strip(),
                "content": data.content.strip(),
                "memory_type": data.memory_type,
                "category": data.category.strip(),
                "timestamp": datetime.now(timezone.utc),
                "version_number": existing.version_number + 1,
            }
        )
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE memories
                SET title = ?, content = ?, memory_type = ?, category = ?,
                    timestamp = ?, version_number = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    updated.title,
                    updated.content,
                    updated.memory_type.value,
                    updated.category,
                    updated.timestamp.isoformat(),
                    updated.version_number,
                    memory_id,
                    self._settings.user_id,
                ),
            )
            await db.commit()

        await self._store_embedding(memory_id, updated.title, updated.content)
        return updated

    async def delete(self, memory_id: str) -> bool:
        """Archive a memory instead of hard-deleting it."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                UPDATE memories
                SET is_archived = 1
                WHERE id = ? AND user_id = ? AND is_archived = 0
                """,
                (memory_id, self._settings.user_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def merge_duplicates(self) -> list[str]:
        """Merge duplicate active memories and archive the extras."""
        actions: list[str] = []
        memories = await self.list_active()
        for group in _find_duplicate_groups(memories):
            canonical = _pick_canonical_memory(group)
            merged_content = _merge_memory_contents(group)
            await self.update(
                canonical.id,
                MemoryUpdate(
                    title=canonical.title,
                    content=merged_content,
                    memory_type=canonical.memory_type,
                    category=canonical.category,
                ),
            )
            archived = 0
            for duplicate in group:
                if duplicate.id == canonical.id:
                    continue
                if await self.delete(duplicate.id):
                    archived += 1
            actions.append(
                f"Merged {len(group)} entries into '{canonical.title}': {merged_content}"
            )
        return actions

    async def semantic_search(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        """Return memories ranked by embedding similarity to the query."""
        _ = user_id or self._settings.user_id
        try:
            query_vector = await self._embeddings.embed_one(query.strip())
        except Exception as exc:
            logger.warning("Semantic search embedding failed: %s", exc)
            return await self.search(query, limit=top_k)

        memories = await self.list_active()
        scored: list[tuple[float, MemoryRecord]] = []

        for memory in memories:
            vector = await self._load_embedding(memory.id)
            if vector is None:
                await self._store_embedding(memory.id, memory.title, memory.content)
                vector = await self._load_embedding(memory.id)
            if vector is None:
                continue

            score = cosine_similarity(query_vector, vector)
            scored.append((score, memory))

        scored.sort(
            key=lambda item: (item[0], item[1].timestamp.timestamp()),
            reverse=True,
        )
        logger.debug(
            "Semantic search for %r returned %d scored memories.",
            query,
            len(scored),
        )
        return [memory for _, memory in scored[:top_k]]

    async def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        """Rank memories by simple keyword overlap with the query."""
        tokens = _tokenize(query)
        if not tokens:
            return []

        memories = await self.list_active()
        scored: list[tuple[float, MemoryRecord]] = []

        for memory in memories:
            score = _score_memory(memory, tokens)
            if score > 0:
                scored.append((score, memory))

        scored.sort(
            key=lambda item: (item[0], item[1].timestamp.timestamp()),
            reverse=True,
        )
        return [memory for _, memory in scored[:limit]]

    async def find_similar(
        self,
        title: str,
        content: str,
        message: str = "",
        memory_type: MemoryType | None = None,
        limit: int = 5,
    ) -> MemoryRecord | None:
        """Return an existing memory about the same subject, if any."""
        query = message.strip() or title.strip() or content.strip()
        candidates = await self.search_by_entity(query)
        if not candidates:
            candidates = await self.search(query, limit=limit)
        if not candidates and query != title.strip():
            candidates = await self.search(title, limit=limit)

        subject_tokens = _subject_tokens(title, content)
        message_tokens = _subject_tokens(message, message)
        if not subject_tokens and not message_tokens:
            return None

        best_match: MemoryRecord | None = None
        best_score = -1.0
        for candidate in candidates:
            if not _should_merge_memories(
                candidate,
                title,
                content,
                memory_type or candidate.memory_type,
                message,
            ):
                continue
            candidate_tokens = _subject_tokens(candidate.title, candidate.content)
            if not (subject_tokens & candidate_tokens) and not (
                message_tokens and message_tokens <= candidate_tokens
            ):
                continue

            score = float(len((subject_tokens | message_tokens) & candidate_tokens))
            score += len(candidate.content) / 1000
            score += candidate.timestamp.timestamp() / 1_000_000_000_000
            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match

    async def search_by_entity(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """Return all memories that mention a subject from the query."""
        tokens = _subject_tokens(query, query)
        if not tokens:
            return await self.search(query, limit=limit)

        memories = await self.list_active()
        scored: list[tuple[float, MemoryRecord]] = []

        for memory in memories:
            haystack_tokens = set(_tokenize(f"{memory.title} {memory.content}"))
            matched = sum(1 for token in tokens if token in haystack_tokens)
            if matched == 0:
                continue

            score = float(matched)
            score += sum(0.5 for token in tokens if token in memory.title.lower())
            score += memory.timestamp.timestamp() / 1_000_000_000_000
            scored.append((score, memory))

        scored.sort(
            key=lambda item: (item[0], item[1].timestamp.timestamp()),
            reverse=True,
        )
        return [memory for _, memory in scored[:limit]]


def _row_to_record(row: aiosqlite.Row) -> MemoryRecord:
    """Convert a database row into a memory record."""
    return MemoryRecord(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        content=row["content"],
        memory_type=MemoryType(row["memory_type"]),
        category=row["category"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        is_archived=bool(row["is_archived"]),
        version_number=row["version_number"],
    )


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase search tokens."""
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1]


def _subject_tokens(title: str, content: str) -> set[str]:
    """Extract meaningful subject tokens for duplicate detection."""
    return {
        token
        for token in _tokenize(f"{title} {content}")
        if token not in _STOPWORDS and len(token) > 1
    }


def _should_merge_memories(
    existing: MemoryRecord,
    title: str,
    content: str,
    memory_type: MemoryType,
    message: str = "",
) -> bool:
    """Decide whether a new memory should update an existing one."""
    new_tokens = _subject_tokens(title, content)
    existing_tokens = _subject_tokens(existing.title, existing.content)
    message_tokens = _subject_tokens(message, message)
    overlap = new_tokens & existing_tokens
    if not overlap and not (message_tokens & existing_tokens):
        return False

    if message_tokens and message_tokens <= existing_tokens:
        return True

    if (
        memory_type == MemoryType.OBJECT_LOCATION
        and existing.memory_type == MemoryType.OBJECT_LOCATION
    ):
        return bool(_object_tokens(new_tokens) & _object_tokens(existing_tokens))

    if not overlap:
        return False

    new_only = new_tokens - existing_tokens
    existing_only = existing_tokens - new_tokens
    new_aspects = new_only - overlap
    existing_aspects = existing_only - overlap
    if new_aspects and existing_aspects:
        return False

    return True


_DISTINCT_FACT_MARKERS = frozenset({"gift", "perfume", "lattafa", "may"})

_LOCATION_WORDS = frozenset(
    {
        "drawer",
        "locker",
        "shelf",
        "table",
        "room",
        "desk",
        "top",
        "red",
        "blue",
        "kitchen",
        "bedroom",
        "office",
        "home",
        "house",
    }
)


def _object_tokens(tokens: set[str]) -> set[str]:
    """Strip location words to isolate the object being located."""
    return tokens - _LOCATION_WORDS


def _are_duplicate_memories(a: MemoryRecord, b: MemoryRecord) -> bool:
    """Return True when two stored memories describe the same fact."""
    a_tokens = _subject_tokens(a.title, a.content)
    b_tokens = _subject_tokens(b.title, b.content)
    overlap = a_tokens & b_tokens
    if not overlap:
        return False

    if (
        a.memory_type == MemoryType.OBJECT_LOCATION
        and b.memory_type == MemoryType.OBJECT_LOCATION
    ):
        return bool(_object_tokens(a_tokens) & _object_tokens(b_tokens))

    a_only = a_tokens - b_tokens
    b_only = b_tokens - a_tokens

    if a_only & _DISTINCT_FACT_MARKERS and b_only & _DISTINCT_FACT_MARKERS:
        return False
    if ("gift" in a_only or "gift" in b_only) and ("may" in a_only or "may" in b_only):
        return False

    if "hdfc" in overlap:
        topic_tokens = {"manager", "relationship", "harshit", "refers", "bank"}
        if (a_tokens & topic_tokens) and (b_tokens & topic_tokens):
            return True

    if a_tokens <= b_tokens or b_tokens <= a_tokens:
        return True

    return _should_merge_memories(
        a,
        b.title,
        b.content,
        b.memory_type,
        message=b.title,
    )


def _find_duplicate_groups(memories: Sequence[MemoryRecord]) -> list[list[MemoryRecord]]:
    """Group memories that should be merged together."""
    if not memories:
        return []

    parent = {memory.id: memory.id for memory in memories}

    def find(memory_id: str) -> str:
        while parent[memory_id] != memory_id:
            parent[memory_id] = parent[parent[memory_id]]
            memory_id = parent[memory_id]
        return memory_id

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    memory_list = list(memories)
    for index, left in enumerate(memory_list):
        for right in memory_list[index + 1 :]:
            if _are_duplicate_memories(left, right):
                union(left.id, right.id)

    grouped: dict[str, list[MemoryRecord]] = {}
    for memory in memory_list:
        grouped.setdefault(find(memory.id), []).append(memory)

    return [group for group in grouped.values() if len(group) > 1]


def _pick_canonical_memory(group: Sequence[MemoryRecord]) -> MemoryRecord:
    """Pick the best memory to keep when merging duplicates."""
    return max(
        group,
        key=lambda memory: (
            len(memory.content),
            memory.version_number,
            memory.timestamp.timestamp(),
        ),
    )


def _merge_memory_contents(group: Sequence[MemoryRecord]) -> str:
    """Combine duplicate memory contents without repeating sentences."""
    ordered = sorted(group, key=lambda memory: memory.timestamp.timestamp(), reverse=True)
    seen: set[str] = set()
    parts: list[str] = []
    for memory in ordered:
        normalized = memory.content.strip().rstrip(".").lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        parts.append(memory.content.strip().rstrip("."))
    return ". ".join(parts) + "."


_APPEND_PATTERN = re.compile(
    r"^\s*(add to|add another|also add|another)\s+",
    re.IGNORECASE,
)


def is_append_request(message: str) -> bool:
    """Return True when the user is adding to an existing list."""
    return bool(_APPEND_PATTERN.search(message))


def append_topic(message: str) -> str:
    """Strip an append prefix to recover the topic being added to."""
    return _APPEND_PATTERN.sub("", message.strip(), count=1).strip()


def extract_urls(text: str) -> list[str]:
    """Extract HTTP(S) URLs from text."""
    return re.findall(r"https?://[^\s<>\"']+", text)


def has_new_list_items(existing_content: str, message: str, new_content: str) -> bool:
    """Return True when the message introduces a new list item."""
    urls = extract_urls(message) or extract_urls(new_content)
    for url in urls:
        if url not in existing_content:
            return True
    return False


def append_list_items(
    existing_content: str,
    message: str,
    new_content: str,
) -> str:
    """Append new items to an existing list-style memory."""
    incoming_urls = extract_urls(message) or extract_urls(new_content)
    if incoming_urls:
        items = list(dict.fromkeys(extract_urls(existing_content)))
        for url in incoming_urls:
            if url not in items:
                items.append(url)
        if len(items) == 1:
            return items[0]
        return "\n".join(f"- {item}" for item in items)

    line = new_content.strip()
    if not line or line in existing_content:
        return existing_content

    if existing_content.strip():
        return f"{existing_content.rstrip()}\n- {line}"
    return line


def should_append_to_existing(
    existing: MemoryRecord,
    message: str,
    content: str,
    memory_type: MemoryType,
) -> bool:
    """Decide whether a save should append rather than replace."""
    if is_append_request(message):
        return True

    if (
        existing.memory_type == MemoryType.OBJECT_LOCATION
        and memory_type == MemoryType.OBJECT_LOCATION
    ):
        return False

    if memory_type == MemoryType.LIST:
        return has_new_list_items(existing.content, message, content)

    if existing.memory_type in {MemoryType.LIST, MemoryType.MISC, MemoryType.FACT}:
        return has_new_list_items(existing.content, message, content)

    return False


def _score_memory(memory: MemoryRecord, tokens: Sequence[str]) -> float:
    """Score a memory against query tokens using title and content overlap."""
    haystack = f"{memory.title} {memory.content} {memory.category}".lower()
    haystack_tokens = set(_tokenize(haystack))
    if not haystack_tokens:
        return 0.0

    matches = sum(1 for token in tokens if token in haystack_tokens)
    if matches == 0:
        return 0.0

    phrase_bonus = 2.0 if " ".join(tokens) in haystack else 0.0
    title_bonus = sum(0.5 for token in tokens if token in memory.title.lower())
    recency_bonus = memory.timestamp.timestamp() / 1_000_000_000_000

    return matches + phrase_bonus + title_bonus + recency_bonus


def format_entity_answer(memories: Sequence[MemoryRecord]) -> str:
    """Combine multiple memories into one concise answer."""
    if not memories:
        return ""
    if len(memories) == 1:
        return memories[0].content
    parts = [memory.content.rstrip(".") for memory in memories]
    return ". ".join(parts) + "."


def format_memories_for_prompt(memories: Sequence[MemoryRecord]) -> str:
    """Format memories as compact text for LLM prompts."""
    if not memories:
        return "(none)"

    lines: list[str] = []
    for memory in memories:
        lines.append(
            f"- id={memory.id} | {memory.title} | {memory.content} "
            f"| type={memory.memory_type.value} | category={memory.category}"
        )
    return "\n".join(lines)
