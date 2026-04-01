from typing import TypedDict, Any, List
from src.graph.rag.utils import (
    search_with_rerank, get_embeddings
)
from langchain_core.output_parsers import StrOutputParser
from src.graph.prompts.rag_prompts import RAG_ANSWER_PROMPT
from src.graph.models.llm import get_llm, client

# Rag Agent state definition
class RagAgentState(TypedDict, total=False):
    question: str # user question
    namespace: str # query namespace
    rerank_results: Any # Pinecone rerank result
    contexts: List[str] # selected topK text fragments
    answer: str # answer to the user question

# Rag Agent class
class RagAgent:
    def __init__(
        self,
        pc,
        index,
        namespace,
        llm_model="gpt-4.1",
        temperature=0.0,
        max_contexts=3
    ):
        self.pc = pc
        if isinstance(index, str):
            self.index = pc.Index(index)
        else:
            self.index = index
        self.namespace = namespace
        self.max_contexts = max_contexts

        self.llm = get_llm(
            role="rag",
            model_name=llm_model,
            temperature=temperature,
            use_langchain=True,
        )
        self.client = client

        self.answer_prompt = RAG_ANSWER_PROMPT

        self.answer_chain = (
            self.answer_prompt
            | self.llm
            | StrOutputParser()
        )

    def generate_answer(self, question, contexts):
        context_str = "\n\n---\n\n".join(contexts)
        return self.answer_chain.invoke(
            {
                "context": context_str,
                "question": question,
            }
        )

    def retrieve_with_rerank(self, question):
        query_vector = get_embeddings(question, self.client)
        return search_with_rerank(
            pc=self.pc,
            index=self.index,
            query=question,
            query_vector=query_vector,
            namespace=self.namespace,
            vector_top_k=10,
            rerank_top_n=self.max_contexts,
            text_field="text",
        )

    def extract_contexts_from_rerank(self, rerank_result):
        contexts = []
        for item in rerank_result.data:
            doc = item.document
            text = getattr(doc, "text", None)
            if text:
                contexts.append(text)
        return contexts

# create rag agent node
def create_rag_agent_node(
    pc,
    index="bank-docs-index",
    namespace="deposits-en",
    llm_model="gpt-4.1",
    temperature=0.0,
    max_contexts=5):
    agent = RagAgent(
        pc=pc,
        index=index,
        namespace=namespace,
        llm_model=llm_model,
        temperature=temperature,
        max_contexts=max_contexts,
    )

    def rag_agent_node(state: RagAgentState) -> RagAgentState:
        question = state.get("question", "")
        if not question:
            return {**state, "answer": "No question provided."}

        # 1. retrieve with rerank
        rerank_result = agent.retrieve_with_rerank(question)

        # 2. extract topK contexts
        contexts = agent.extract_contexts_from_rerank(rerank_result)

        # 3. LLM generate answer
        answer = agent.generate_answer(question, contexts)

        # 4. write back to state
        return {
            **state,
            "namespace": agent.namespace,
            "rerank_results": rerank_result,
            "contexts": contexts,
            "answer": answer,
        }
    return rag_agent_node
