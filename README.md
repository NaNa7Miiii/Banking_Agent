## Router + SQL Module

This module handles transaction-related queries and routes them to SQL-based data retrieval.

### Key contributions
- Implemented routing from user query → SQL agent
- Improved robustness for date parsing (YYYY-MM-DD)
- Supports both single-day and date-range queries
- Returns user-friendly error messages instead of failing on invalid input

### Related branches
- router-sql-evaluation: midterm work (router + SQL + evaluation)
- router-sql: post-reorg improvements (robustness and edge-case handling)

## Additional Work

### 1. Evaluation

We added an evaluation component to better understand system performance.

This includes:
- basic metrics (accuracy, precision, recall, F1-score)
- confusion matrix
- simple comparison across different settings

The goal is to make the system more interpretable and verify that the results are reasonable.

### 2. Experiment / Analysis

We also performed additional analysis to understand fraud patterns and model behavior.

This includes:
- feature-level observations
- how different signals contribute to fraud detection
- general patterns in transaction data

These experiments help explain why the system works, not just how it is implemented.
