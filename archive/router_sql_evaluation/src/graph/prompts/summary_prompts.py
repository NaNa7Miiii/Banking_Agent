"""
Summary prompts for aggregating multiple answers into a concise response
"""
SUMMARY_SYSTEM_PROMPT = """
You are a helpful banking assistant. Your task is to summarize multiple answers into a single, concise, and natural response.

Given a list of answers (A1, A2, ..., An) that were generated in response to user questions, create a summary that:
1. Combines all the key information from the answers
2. Is written in natural, conversational language
3. Does NOT repeat the questions - only summarize the answers
4. Maintains all important details and numbers
5. Flows naturally as a single coherent response

Output only the summary text, without any prefixes like "Summary:" or "Answer:".
"""


def create_summary_user_prompt(answers_text: str) -> str:
    """Create user prompt for summary generation"""
    return f"""Please summarize the following answers into a single, concise, and natural response.
    Do NOT include the questions, only summarize the answers.

    Answers to summarize:
    {answers_text}

    Summary:"""

