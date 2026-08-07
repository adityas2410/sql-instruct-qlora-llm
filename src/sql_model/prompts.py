"""Prompt templates for Falcon SQL instruction tuning and inference."""

from __future__ import annotations

SQL_SYSTEM_INSTRUCTIONS = """You are an insurance investigation SQL assistant.
Generate exactly one safe read-only PostgreSQL SELECT query.
Use only the approved schema provided in the prompt.
Do not use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, comments, or multiple statements.
Prefer explicit joins through declared foreign keys.
Return SQL only, with no markdown or explanation.
""".strip()

SQL_PROMPT_TEMPLATE = """{system_instructions}

Approved schema:
{schema}

Investigator request:
{instruction}

SQL:
"""

SQL_TRAINING_RESPONSE_TEMPLATE = "{sql}".strip()


def render_sql_prompt(instruction: str, schema: str) -> str:
    """Render a SQL generation prompt for Falcon."""
    return SQL_PROMPT_TEMPLATE.format(
        system_instructions=SQL_SYSTEM_INSTRUCTIONS,
        schema=schema.strip(),
        instruction=instruction.strip(),
    )


def render_training_text(instruction: str, schema: str, sql: str) -> str:
    """Render one supervised fine-tuning text sample."""
    prompt = render_sql_prompt(instruction=instruction, schema=schema)
    return f"{prompt}{SQL_TRAINING_RESPONSE_TEMPLATE.format(sql=sql.strip())}"
