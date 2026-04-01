"""Utility functions for fraud detection agent"""

import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy import text
from src.graph.modules.sql_agent import SQLAgent
from src.graph.models.llm import get_llm
from src.graph.prompts.fraud_prompts import FRAUD_LLM_SYSTEM_PROMPT, FRAUD_ANALYSIS_USER_PROMPT


def extract_transaction_id_from_question(question: str) -> Optional[str]:
    """Extract transaction ID from user question.

    Matches patterns like:
    - "explain transaction XXX"
    - "transaction XXX"
    - "XXX" (only if it's a 32-character hex string, typical transaction ID format)
    """
    question_lower = question.lower().strip()

    # Pattern 1: Explicit transaction ID mentions
    # Matches: "explain transaction XXX", "transaction XXX", etc.
    explicit_pattern = r'(?:explain|why|analyze|details?|info|about|for)\s+(?:transaction\s+)?([a-z0-9]{32})|(?:transaction\s+\s+)([a-z0-9]{32})'
    match = re.search(explicit_pattern, question_lower)
    if match:
        return match.group(1) or match.group(2)

    # Pattern 2: Standalone 32-character hex string (typical transaction ID format)
    # Only match if it's exactly 32 characters of hex, not part of a date or number
    standalone_pattern = r'\b([a-z0-9]{32})\b'
    match = re.search(standalone_pattern, question_lower)
    if match:
        # Make sure it's not part of a date or other number sequence
        trans_id = match.group(1)
        # Check if it's surrounded by spaces or word boundaries (not part of a longer string)
        return trans_id

    return None


def fetch_transaction_by_id(sql_agent: SQLAgent, transaction_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single transaction from database by transaction_id."""
    try:
        fetch_query = text(f"SELECT * FROM {sql_agent.table_name} WHERE transaction_id = :trans_id LIMIT 1")

        with sql_agent.db._engine.connect() as conn:
            result = conn.execute(fetch_query, {"trans_id": transaction_id})
            transaction_rows = [dict(row._mapping) for row in result]

        if transaction_rows:
            return transaction_rows[0]
        return None
    except Exception as e:
        print(f"[Fraud Utils] Error fetching transaction {transaction_id}: {str(e)}")
        return None


def format_transaction_for_llm(transaction: Dict[str, Any], fraud_prob: float, confidence: str) -> str:
    """Format transaction data for LLM analysis."""
    transaction_text = f"Transaction ID: {transaction.get('transaction_id', 'N/A')}\n"
    transaction_text += f"ML Fraud Probability: {fraud_prob:.4f} ({confidence} confidence)\n"
    transaction_text += f"Transaction Details:\n"
    for key, value in transaction.items():
        if value is not None:
            transaction_text += f"  {key}: {value}\n"
    return transaction_text


def get_llm_analysis_for_transaction(
    llm,
    transaction: Dict[str, Any],
    fraud_prob: float,
    confidence: str
) -> tuple[str, List[Dict[str, Any]]]:
    """Get LLM analysis for a single transaction.

    Returns:
        tuple: (raw_llm_response, parsed_json_list)
    """
    transaction_text = format_transaction_for_llm(transaction, fraud_prob, confidence)

    user_prompt_messages = FRAUD_ANALYSIS_USER_PROMPT.format_messages(
        transactions_text=transaction_text
    )
    user_prompt = user_prompt_messages[0].content if user_prompt_messages else transaction_text

    llm_analysis_raw = llm.chat(
        system_prompt=FRAUD_LLM_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_format=None,
    )

    # Parse JSON response
    llm_analysis_json = []
    try:
        cleaned = llm_analysis_raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        llm_analysis_json = json.loads(cleaned)
        if not isinstance(llm_analysis_json, list):
            llm_analysis_json = [llm_analysis_json] if isinstance(llm_analysis_json, dict) else []
    except (json.JSONDecodeError, Exception) as e:
        print(f"[Fraud Utils] Warning: Failed to parse LLM JSON response: {e}")
        llm_analysis_json = []

    return llm_analysis_raw, llm_analysis_json


def format_llm_explanation_report(
    transaction_id: str,
    transaction: Dict[str, Any],
    fraud_prob: float,
    confidence: str,
    llm_analysis_json: List[Dict[str, Any]],
    llm_analysis_raw: str
) -> str:
    """Format LLM explanation report for a single transaction."""
    explanation_parts = [
        f"LLM Analysis for Transaction: {transaction_id}",
        f"=" * 60,
        f"",
        f"ML Model Prediction:",
        f"  Fraud Probability: {fraud_prob:.4f} ({confidence} confidence)",
        f"",
        f"Transaction Details:",
        f"  Amount: ${transaction.get('transaction_amount', 'N/A')}",
        f"  Merchant: {transaction.get('merchant_name', 'N/A')}",
        f"  Category: {transaction.get('merchant_category', 'N/A')}",
        f"  Date: {transaction.get('transaction_datetime', 'N/A')}",
        f"",
        f"AI Explanation:",
        f"-" * 60,
    ]

    if llm_analysis_json:
        for analysis in llm_analysis_json:
            explanation = analysis.get("Explanation", "No explanation provided.")
            explanation_parts.append(explanation)
            explanation_parts.append("")
    else:
        explanation_parts.append(llm_analysis_raw)

    return "\n".join(explanation_parts)


def extract_time_window_from_transactions(transactions: List[Dict[str, Any]]) -> tuple[str, str]:
    """Extract time window (start, end) from transactions."""
    time_window_start = ""
    time_window_end = ""

    if transactions:
        transaction_dates = [t.get("transaction_datetime") for t in transactions if t.get("transaction_datetime")]
        if transaction_dates:
            try:
                dates = [
                    datetime.fromisoformat(str(d).replace("Z", "+00:00")) if isinstance(d, str) else d
                    for d in transaction_dates if d
                ]
                if dates:
                    time_window_start = (
                        min(dates).strftime("%Y-%m-%d")
                        if hasattr(min(dates), 'strftime')
                        else str(min(dates)).split()[0]
                    )
                    time_window_end = (
                        max(dates).strftime("%Y-%m-%d")
                        if hasattr(max(dates), 'strftime')
                        else str(max(dates)).split()[0]
                    )
            except Exception:
                pass

    return time_window_start, time_window_end


def format_fraud_detection_report(
    total_transactions: int,
    flagged_count: int,
    threshold: float,
    transaction_ids: List[str]
) -> str:
    """Format fraud detection report with flagged transaction IDs."""
    report_parts = [
        f"Fraud Detection Results",
        f"=" * 60,
        f"",
        f"Total transactions analyzed: {total_transactions}",
        f"Flagged transactions (fraud probability >= {threshold:.4f}): {flagged_count}",
        f"",
        f"Flagged Transaction IDs:",
        f"-" * 60,
    ]

    # List all transaction IDs
    for idx, trans_id in enumerate(transaction_ids, 1):
        report_parts.append(f"{idx}. {trans_id}")

    return "\n".join(report_parts)


def format_no_fraud_report(total_transactions: int, threshold: float) -> str:
    """Format report when no transactions are flagged."""
    return (
        f"Fraud Detection Analysis Report\n"
        f"{'=' * 50}\n\n"
        f"Total transactions analyzed: {total_transactions}\n"
        f"Transactions flagged by ML model (probability >= {threshold:.2f}): 0\n\n"
        f"Result: No suspicious transactions detected.\n"
        f"All transactions have fraud probability below the threshold ({threshold:.2f})."
    )
