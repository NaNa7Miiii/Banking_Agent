import json
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path
from sqlalchemy import text
from src.data.model.predictor import get_fraud_predictor
from src.graph.models.llm import get_llm
from src.graph.modules.sql_agent import SQLAgent
from src.graph.utils.fraud_utils import (
    extract_transaction_id_from_question,
    fetch_transaction_by_id,
    get_llm_analysis_for_transaction,
    format_llm_explanation_report,
    extract_time_window_from_transactions,
    format_fraud_detection_report,
    format_no_fraud_report,
)


# Fraud agent state definition
class FraudAgentState(TypedDict):
    question: str  # User's question about fraud detection
    transactions: List[Dict[str, Any]]  # List of transaction dictionaries (can be provided directly or fetched from DB)
    sql_query: str  # SQL query generated to fetch transactions
    sql_result: Any  # Raw SQL query result
    fraud_predictions: List[Dict[str, Any]]  # ML model predictions for each transaction
    flagged_transactions: List[Dict[str, Any]]  # Transactions with fraud_probability >= threshold
    llm_analysis: str  # LLM's final fraud analysis (raw JSON string)
    llm_analysis_json: List[Dict[str, Any]]  # Parsed LLM analysis as JSON
    final_fraud_report: str  # Combined report of fraud analysis
    answer: str  # Formatted answer for router (compatible with router interface)
    time_window_start: str  # Start date of time window
    time_window_end: str  # End date of time window

# This agent handles two main functions:
#    1. Single transaction detection: SQL (fetch metadata) → XGBoost → LLM → Output explanation
#    2. Time window detection: SQL (fetch transactions) → XGBoost → Output flagged transaction IDs
def create_fraud_agent_node(
    sql_agent: Optional[SQLAgent] = None,
    llm_model="gpt-4.1",
    temperature=0.1,
    model_dir: Optional[Path] = None
):
    # Load required columns from model info
    if model_dir is None:
        model_dir = Path(__file__).parent.parent.parent.parent / "src" / "data" / "model"
    else:
        model_dir = Path(model_dir)

    model_info_path = model_dir / "model_info.json"
    required_columns = None
    if model_info_path.exists():
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
            # Get base columns needed (excluding derived features that will be computed in predictor)
            feature_cols = model_info.get("feature_cols", [])
            base_columns_needed = set()
            for col in feature_cols:
                # Derived features that need base columns
                if col in ["trans_hour", "trans_dayofweek", "trans_day", "trans_month", "is_night"]:
                    base_columns_needed.add("transaction_datetime")
                elif col == "customer_age":
                    base_columns_needed.add("customer_dob")
                else:
                    # Direct column from DB
                    base_columns_needed.add(col)
            base_columns_needed.add("transaction_id")
            # Explicitly exclude is_fraud - it's a forbidden column that should never be queried
            base_columns_needed.discard("is_fraud")
            required_columns = sorted(list(base_columns_needed))

    predictor = get_fraud_predictor(model_dir)

    # If sql_agent is provided but doesn't have required_columns, create a new one
    if sql_agent and required_columns:
        # Create a new SQLAgent with required_columns to ensure SQL queries include all necessary columns
        sql_agent = SQLAgent(
            db=sql_agent.db,
            table_name=sql_agent.table_name,
            llm_model=llm_model,
            temperature=0.0,  # Use 0.0 for SQL generation
            required_columns=required_columns,
        )

    # Create LLM instance using factory function
    llm = get_llm(
        role="fraud",
        model_name=llm_model,
        temperature=temperature,
    )

    def _handle_single_transaction_detection(
        state: Dict[str, Any],
        transaction_id: str,
        sql_agent: SQLAgent,
        predictor,
        llm
    ) -> Dict[str, Any]:
        """Handle single transaction detection: SQL → XGBoost → LLM → Output explanation."""
        print(f"\n[Fraud Agent] Single transaction detection for ID: {transaction_id}")

        # Step 1: SQL agent - Fetch transaction metadata
        print("[Fraud Agent] Step 1: Fetching transaction metadata from database...")
        transaction = fetch_transaction_by_id(sql_agent, transaction_id)
        if not transaction:
            return {
                **state,
                "answer": f"Transaction ID '{transaction_id}' not found in the database.",
                "llm_analysis": "",
                "llm_analysis_json": [],
                "final_fraud_report": f"Transaction ID '{transaction_id}' not found.",
            }

        # Step 2: XGBoost prediction
        print("[Fraud Agent] Step 2: Running XGBoost model for fraud prediction...")
        fraud_prediction = predictor.predict(transaction)
        fraud_prob = fraud_prediction.get("fraud_probability", 0.0)
        confidence = fraud_prediction.get("confidence", "unknown")
        print(f"[Fraud Agent] XGBoost prediction: fraud_probability={fraud_prob:.4f}, confidence={confidence}")

        # Step 3: LLM analysis
        print("[Fraud Agent] Step 3: Requesting LLM explanation...")
        llm_analysis_raw, llm_analysis_json = get_llm_analysis_for_transaction(
            llm, transaction, fraud_prob, confidence
        )
        print("[Fraud Agent] LLM explanation received.")

        # Format explanation report
        answer = format_llm_explanation_report(
            transaction_id, transaction, fraud_prob, confidence,
            llm_analysis_json, llm_analysis_raw
        )

        time_window_start, time_window_end = extract_time_window_from_transactions([transaction])

        return {
            **state,
            "transactions": [transaction],
            "fraud_predictions": [fraud_prediction],
            "flagged_transactions": [{
                "transaction": transaction,
                "fraud_probability": fraud_prob,
                "confidence": confidence,
                "ml_is_fraud": fraud_prediction.get("is_fraud", False),
            }],
            "llm_analysis": llm_analysis_raw,
            "llm_analysis_json": llm_analysis_json,
            "final_fraud_report": answer,
            "answer": answer,
            "time_window_start": time_window_start,
            "time_window_end": time_window_end,
        }

    def _handle_time_window_detection(
        state: Dict[str, Any],
        question: str,
        sql_agent: SQLAgent,
        predictor
    ) -> Dict[str, Any]:
        """Handle time window fraud detection: SQL → XGBoost → Output flagged transaction IDs."""
        try:
            print(f"\n[Fraud Agent] Time window detection: Generating SQL query from question: {question}")
            sql_query = sql_agent.generate_sql(question)
            print(f"[Fraud Agent] SQL Query: {sql_query}")

            print("[Fraud Agent] Executing SQL query...")
            with sql_agent.db._engine.connect() as conn:
                result = conn.execute(text(sql_query))
                transactions = [dict(row._mapping) for row in result]
                sql_result = transactions
            print(f"[Fraud Agent] SQL query executed. Retrieved {len(transactions)} transactions.")

            if not transactions:
                return {
                    **state,
                    "sql_query": sql_query,
                    "sql_result": sql_result,
                    "transactions": [],
                    "fraud_predictions": [],
                    "flagged_transactions": [],
                    "llm_analysis": "",
                    "llm_analysis_json": [],
                    "final_fraud_report": "No transactions found matching the query criteria.",
                    "answer": "No transactions found matching the query criteria.",
                    "time_window_start": "",
                    "time_window_end": "",
                }

            # Run XGBoost detection and output flagged transaction IDs
            return _handle_batch_fraud_detection(
                state, transactions, sql_query, sql_result, predictor
            )
        except Exception as e:
            print(f"[Fraud Agent] ERROR: Failed to fetch transactions: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                **state,
                "sql_query": state.get("sql_query", ""),
                "sql_result": None,
                "transactions": [],
                "fraud_predictions": [],
                "flagged_transactions": [],
                "llm_analysis": "",
                "llm_analysis_json": [],
                "final_fraud_report": f"Error fetching transactions: {str(e)}",
                "answer": f"Error fetching transactions: {str(e)}",
                "time_window_start": "",
                "time_window_end": "",
            }

    def _handle_batch_fraud_detection(
        state: Dict[str, Any],
        transactions: List[Dict[str, Any]],
        sql_query: str,
        sql_result: Any,
        predictor
    ) -> Dict[str, Any]:
        """Handle batch fraud detection: XGBoost prediction and report generation."""
        # Extract time window from transactions
        time_window_start, time_window_end = extract_time_window_from_transactions(transactions)

        # Stage 1: ML model predictions for all transactions
        print("\n[Fraud Agent] Stage 1: Model Detection - Running XGBoost model on transactions...")
        print(f"[Fraud Agent] Analyzing {len(transactions)} transactions...")
        fraud_predictions = predictor.predict_batch(transactions)
        print(f"[Fraud Agent] Model Detection completed. Processed {len(fraud_predictions)} predictions.")

        # Filter transactions with fraud_probability >= threshold
        threshold = predictor.decision_threshold
        print(f"[Fraud Agent] Filtering transactions with fraud probability >= {threshold:.4f}...")
        flagged_transactions = []

        for transaction, prediction in zip(transactions, fraud_predictions):
            fraud_prob = prediction.get("fraud_probability", 0.0)
            if fraud_prob >= threshold:
                flagged_transactions.append({
                    "transaction": transaction,
                    "fraud_probability": fraud_prob,
                    "confidence": prediction.get("confidence", "unknown"),
                    "ml_is_fraud": prediction.get("is_fraud", False),
                })

        print(f"[Fraud Agent] Found {len(flagged_transactions)} transactions exceeding threshold.")

        # Generate report with flagged transaction IDs
        if flagged_transactions:
            print("[Fraud Agent] Generating fraud detection report with flagged transactions...")

            # Extract transaction IDs
            transaction_ids = [
                item["transaction"].get("transaction_id")
                for item in flagged_transactions
                if item["transaction"].get("transaction_id")
            ]

            # Format report
            final_fraud_report = format_fraud_detection_report(
                len(transactions), len(flagged_transactions), threshold, transaction_ids
            )

            answer = final_fraud_report
            print("[Fraud Agent] Fraud detection analysis completed. LLM explanation available on request.")
        else:
            print("[Fraud Agent] No transactions exceeded the fraud probability threshold.")
            final_fraud_report = format_no_fraud_report(len(transactions), threshold)
            answer = final_fraud_report

        return {
            **state,
            "sql_query": sql_query,
            "sql_result": sql_result,
            "transactions": transactions,
            "fraud_predictions": fraud_predictions,
            "flagged_transactions": flagged_transactions,
            "llm_analysis": "",
            "llm_analysis_json": [],
            "final_fraud_report": final_fraud_report,
            "answer": answer,
            "time_window_start": time_window_start,
            "time_window_end": time_window_end,
        }

    def fraud_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Fraud agent node with two main functions:
        1. Single transaction detection: SQL → XGBoost → LLM → Output explanation
        2. Time window detection: SQL → XGBoost → Output flagged transaction IDs
        """
        question = state.get("question", "")
        transactions = state.get("transactions", [])

        if not sql_agent:
            return {
                **state,
                "fraud_predictions": [],
                "flagged_transactions": [],
                "llm_analysis": "",
                "llm_analysis_json": [],
                "final_fraud_report": "SQL agent not available.",
                "answer": "SQL agent not available.",
                "time_window_start": "",
                "time_window_end": "",
            }

        # Case 1: Single transaction detection (SQL → XGBoost → LLM)
        transaction_id = extract_transaction_id_from_question(question)
        if transaction_id:
            return _handle_single_transaction_detection(
                state, transaction_id, sql_agent, predictor, llm
            )

        # Case 2: Time window fraud detection (SQL → XGBoost → Output IDs)
        # If transactions already provided, use them directly
        if transactions:
            return _handle_batch_fraud_detection(
                state, transactions, state.get("sql_query", ""), state.get("sql_result"), predictor
            )

        # If question provided but no transactions, generate SQL and fetch
        if question:
            return _handle_time_window_detection(state, question, sql_agent, predictor)

        # No valid input
        return {
            **state,
            "fraud_predictions": [],
            "flagged_transactions": [],
            "llm_analysis": "",
            "llm_analysis_json": [],
            "final_fraud_report": "No transactions provided for fraud detection.",
            "answer": "No transactions provided for fraud detection.",
            "time_window_start": "",
            "time_window_end": "",
        }


    return fraud_agent_node


# Creates a fraud agent node for the transactions table (similar to create_transaction_sql_agent_node)
def create_transaction_fraud_agent_node(
    username,
    password,
    host,
    port,
    database,
    table_name="transactions",
    llm_model="gpt-4.1",
    temperature=0.1,
    model_dir: Optional[Path] = None):
    """Create a fraud agent node that can be integrated into the main graph"""
    from src.data.utils import get_sql_db

    db = get_sql_db(
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        echo=False,
    )

    # Create SQL agent for fraud agent to use
    sql_agent = SQLAgent(
        db=db,
        table_name=table_name,
        llm_model=llm_model,
        temperature=0.0,  # Use 0.0 for SQL generation
    )

    return create_fraud_agent_node(
        sql_agent=sql_agent,
        llm_model=llm_model,
        temperature=temperature,
        model_dir=model_dir,
    )

