"""Character ids are formatted into table names, so they must be reduced to [a-z0-9_]."""
import pytest

from core.identifiers import safe_char_id
from proxy.api.routes import ChatCompletionMessage, _extract_target_char


@pytest.mark.parametrize("raw, expected", [
    ("Arvenia", "arvenia"),
    ("Mira Vale", "mira_vale"),
    ("Jean-Luc", "jean_luc"),
    ("  Seraphina  ", "seraphina"),
    ("x; DROP TABLE csa_memory_luna; --", "x_drop_table_csa_memory_luna"),
    ("", "default"),
    ("!!!", "default"),
    ("a" * 100, "a" * 48),
])
def test_safe_char_id(raw, expected):
    assert safe_char_id(raw) == expected


def test_message_name_cannot_inject_sql():
    msgs = [ChatCompletionMessage(role="user", content="hi", name="luna; DROP TABLE csa_memory_luna;--")]
    char = _extract_target_char(msgs)
    assert char == "luna_drop_table_csa_memory_luna"
    assert all(c.isalnum() or c == "_" for c in char)


def test_multi_word_character_card_gives_valid_table_suffix():
    msgs = [ChatCompletionMessage(role="system", content="[Character: Mira Vale]\nA tavern keeper.")]
    assert _extract_target_char(msgs) == "mira_vale"


@pytest.mark.asyncio
async def test_multi_word_character_memories_round_trip():
    """Before the fix, 'mira vale' produced an unquoted table name with a space: every insert failed."""
    import asyncpg
    from proxy.rag.retriever import EpisodicRAGRetriever
    from tests._testdb import TEST_DB_CONFIG

    db_pool = await asyncpg.create_pool(**TEST_DB_CONFIG)

    char = _extract_target_char([ChatCompletionMessage(role="system", content="[Character: Mira Vale]")])
    async with db_pool.acquire() as conn:
        await conn.execute("SELECT create_csa_memory_table($1);", char)
        vec = "[" + ",".join(["0.1"] * 3584) + "]"
        await conn.execute(
            f"INSERT INTO csa_memory_{char} (session_id, sensory_input, episodic_embedding) VALUES ('s', 'hello', $1::vector)",
            vec,
        )
    found = await EpisodicRAGRetriever(db_pool).retrieve_memories(
        character_id="Mira Vale", query_embedding=[0.1] * 3584, session_id="s"
    )
    await db_pool.close()
    assert [m["sensory_input"] for m in found] == ["hello"]
