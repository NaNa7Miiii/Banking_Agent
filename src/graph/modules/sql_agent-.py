from typing import TypedDict, Any
from sqlalchemy import inspect
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase

from src.data.utils import get_sql_db
from src.graph.prompts.sql_prompts import create_sql_prompt, create_answer_prompt
from src.graph.models.llm import get_llm

# SQL Agent state definition
class SQLAgentState(TypedDict):
    question: str
    schema: str
    query: str
    response: Any
    answer: str

# format schema from inspector
def format_schema_from_inspector(columns_info, table_name):
    lines = [f"Table: {table_name}", "-" * (7 + len(table_name))]
    for col in columns_info:
        name = col["name"]
        col_type = str(col["type"])
        nullable = col.get("nullable", True)
        nullable_str = "NULL" if nullable else "NOT NULL"
        default = col.get("default")
        default_str = f", DEFAULT {default}" if default is not None else ""
        lines.append(f"{name} {col_type} {nullable_str}{default_str}")
    return "\n".join(lines)

# get db schema
def get_db_schema(db, table_name):
    inspector = inspect(db._engine)
    columns_info = inspector.get_columns(table_name)
    return format_schema_from_inspector(columns_info, table_name)

# sql agent class that encapsulates sql query generation and execution logic
class SQLAgent:
    def __init__(
        self,
        db,
        table_name,
        llm_model="gpt-4.1",
        temperature=0.0,
        required_columns=None):
        self.db = db
        self.table_name = table_name
        self.llm = get_llm(
            role="sql",
            model_name=llm_model,
            temperature=temperature,
            use_langchain=True,
        )

        # Get schema
        self.schema_text = get_db_schema(db, table_name)

        # Create prompts with optional required columns
        self.sql_prompt = create_sql_prompt(required_columns=required_columns)
        self.answer_prompt = create_answer_prompt()

        # Create SQL chain
        self.sql_chain = (
            RunnablePassthrough.assign(schema=lambda _: self.schema_text)
            | self.sql_prompt
            | self.llm.bind(stop="\nSQL Result:")
            | StrOutputParser()
        )

        # Create full chain
        self.full_chain = (
            RunnablePassthrough.assign(
                schema=lambda _: self.schema_text,
                query=self.sql_chain,
            )
            .assign(
                response=lambda variables: self.db.run(variables["query"]),
            )
            | self.answer_prompt
            | self.llm
            | StrOutputParser()
        )

    def run_query(self, query):
        return self.db.run(query)

    def generate_sql(self, question):
        return self.sql_chain.invoke({"question": question})

    def invoke(self, question):
        return self.full_chain.invoke({"question": question})


# general sql agent node
def create_sql_agent_node(
    db,
    table_name,
    llm_model="gpt-4.1",
    temperature=0.0):
    agent = SQLAgent(db, table_name, llm_model, temperature)

    def sql_agent_node(state):
        question = state.get("question", "")
        if not question:
            return {**state, "answer": "No question provided."}

        # Execute full pipeline
        answer = agent.invoke(question)

        sql_query = agent.generate_sql(question)
        sql_result = agent.run_query(sql_query)

        return {
            **state,
            "schema": agent.schema_text,
            "query": sql_query,
            "response": sql_result,
            "answer": answer,
        }

    return sql_agent_node


# creates a sql agent node for the transactions table
def create_transaction_sql_agent_node(
    username,
    password,
    host,
    port,
    database,
    table_name="transactions",
    llm_model="gpt-4.1",
    temperature=0.0):
    db = get_sql_db(
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        echo=False,
    )
    return create_sql_agent_node(db, table_name, llm_model, temperature)

