from typing import TypedDict, Any, List
from langchain_core.output_parsers import StrOutputParser

from src.graph.rag.tavily_search import tavily_search, extract_contexts_from_tavily
from src.graph.prompts.rag_prompts import TAVILY_RAG_ANSWER_PROMPT
from src.graph.models.llm import get_llm

class TavilyAgentState(TypedDict, total=False):
    question: str
    tavily_raw_result: Any
    contexts: List[str]
    sources: List[dict]
    answer: str

class TavilyAgent:
    def __init__(
        self,
        tavily_client,
        llm_model="gpt-4.1",
        temperature=0.0,
        max_contexts=10,
        topic="finance",
        search_depth="advanced",
        max_results=10):

        self.tavily_client = tavily_client
        self.max_contexts = max_contexts
        self.topic = topic
        self.search_depth = search_depth
        self.max_results = max_results

        self.llm = get_llm(
            role="tavily",
            model_name=llm_model,
            temperature=temperature,
            use_langchain=True,
        )

        self.answer_prompt = TAVILY_RAG_ANSWER_PROMPT

        self.answer_chain = (
            self.answer_prompt
            | self.llm
            | StrOutputParser()
        )

    def search_with_tavily(self, question):
        res = tavily_search(
            client=self.tavily_client,
            query=question,
            search_depth=self.search_depth,
            include_answer=False,
            include_images=False,
            include_raw_content=False,
            max_results=self.max_results,
            topic=self.topic,
        )
        return res

    def generate_answer(self, question, contexts):
        if not contexts:
            context_str = "No web search context was retrieved."
        else:
            context_str = "\n\n---\n\n".join(contexts)

        return self.answer_chain.invoke(
            {
                "context": context_str,
                "question": question,
            }
        )

def create_tavily_agent_node(
    tavily_client,
    llm_model="gpt-4.1",
    temperature=0.0,
    max_contexts=10,
    topic="finance",
    search_depth="advanced",
    max_results=10):

    agent = TavilyAgent(
        tavily_client=tavily_client,
        llm_model=llm_model,
        temperature=temperature,
        max_contexts=max_contexts,
        topic=topic,
        search_depth=search_depth,
        max_results=max_results,
    )

    def tavily_agent_node(state: TavilyAgentState) -> TavilyAgentState:
        question = state.get("question", "")
        if not question:
            return {**state, "answer": "No question provided."}

        tavily_result = agent.search_with_tavily(question)
        contexts, sources = extract_contexts_from_tavily(tavily_result, max_contexts=agent.max_contexts)
        answer = agent.generate_answer(question, contexts)

        return {
            **state,
            "tavily_raw_result": tavily_result,
            "contexts": contexts,
            "sources": sources,
            "answer": answer,
        }

    return tavily_agent_node



