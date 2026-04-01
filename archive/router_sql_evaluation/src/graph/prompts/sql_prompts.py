"""
SQL-related prompts for SQL Agent
Contains prompt templates for SQL generation and natural language answer generation
"""
from langchain_core.prompts import ChatPromptTemplate


def create_sql_prompt(required_columns=None) -> ChatPromptTemplate:
    """Create prompt template for SQL query generation

    Args:
        required_columns: Optional list of column names that must be included in the SELECT clause.
                         If provided, these columns will be added to the prompt instructions.
    """
    required_columns_section = ""
    if required_columns:
        required_columns_str = ", ".join(required_columns)
        required_columns_section = f"""

    5. **Required columns (CRITICAL)**
      - You **MUST** include ALL of the following columns in your SELECT clause:
        {required_columns_str}
      - These columns are required for downstream processing and cannot be omitted.
      - You may include additional columns if needed, but these are mandatory.
"""

    template = f"""
    You are an expert PostgreSQL data analyst and SQL engineer.

    Your task is to write a **single, syntactically correct SQL query** that answers the user's question,
    using only the tables and columns described in the schema below.

    ---------------- SCHEMA (important) ----------------
    {{schema}}
    ----------------------------------------------------

    Follow these rules carefully:

    1. **Output format**
      - Return **only** the SQL query, nothing else.
      - Do **not** include backticks, markdown fences, comments, or explanations.
      - Do not include the word "SQL" or "Query" outside of the statement itself.

    2. **Use of schema**
      - Only reference tables and columns that exist in the schema above.
      - If the question is ambiguous, make the **most reasonable assumption** based on column names.
      - Prefer explicit column lists instead of `SELECT *` in production-style queries,
        unless the question explicitly asks for "all columns".

    3. **Safety**
      - Do **not** modify data: no `INSERT`, `UPDATE`, `DELETE`, `DROP`, or `ALTER`.
      - Only generate `SELECT` queries.

    4. **Forbidden columns (CRITICAL)**
      - You **MUST NOT** include the `is_fraud` column in your SELECT clause.
      - You **MUST NOT** use `is_fraud` in your WHERE clause or any other part of the query.
      - The `is_fraud` column is forbidden and should never be referenced in any way.{required_columns_section}

    ---------------- QUESTION ----------------
    {{question}}
    ------------------------------------------

    Now produce the final SQL query (PostgreSQL dialect), and **only** the query:
    """
    return ChatPromptTemplate.from_template(template)


def create_answer_prompt() -> ChatPromptTemplate:
    """Create prompt template for natural language answer generation"""
    template = """
    You are a data analyst.

    Based on the schema, the user's question, the SQL query, and the SQL result,
    give a short, clear explanation in natural language.

    - Speak to a non-technical business user.
    - Use only the information in the SQL result.
    - If the result is empty, say that no relevant data was found.
    - Do not show or explain the SQL query itself.
    - If the question references previous conversation context, use that context to provide a more relevant answer.

    Schema:
    {schema}

    Question:
    {question}

    SQL result:
    {response}
    """
    return ChatPromptTemplate.from_template(template)

