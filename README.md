# Router + SQL Module (Midterm Version)

This branch contains the Router + SQL work developed during the midterm stage of the project.

The main focus of this version is not only functionality, but also reliability, interpretability, and safety in financial query handling.


## What this module does

The Router + SQL module is responsible for:

- classifying user queries into predefined intent types
- routing transaction-related questions to the SQL component
- retrieving structured financial data from the database


## Main design focus

This version treats the Router + SQL component as a problem that can be analyzed and evaluated, rather than only as an engineering feature.

### 1. Routing as an intent classification task
- The router is used to classify user questions into structured intent categories
- The output is normalized into a consistent query format
- The design emphasizes interpretability and error analysis, not only execution flow

### 2. Reliability and evaluation
- Routing performance is evaluated using standard classification metrics
- The evaluation includes:
  - accuracy
  - precision / recall / F1-score
  - confusion matrix
- Additional attention is given to ambiguous queries and misclassified cases

### 3. SQL safety and control
- The SQL agent is designed to remain read-only
- Query generation is constrained by:
  - SELECT-only validation
  - enforced LIMIT
  - filtering of unsafe statements
- These constraints make the SQL component more suitable for financial data access, where safety and verification are important


## Key components

### Router
- Converts user input into structured intent labels
- Produces normalized query objects
- Handles malformed or unexpected LLM outputs safely

### SQL Agent
- Generates SQL queries from natural language questions
- Applies safety checks before execution
- Includes fallback behavior when SQL generation fails


## Evaluation

This branch includes an evaluation pipeline for the routing component.

The evaluation focuses on:
- classification performance across intent types
- confusion matrix analysis
- comparison of routing behavior under different query cases

This allows the Router to be examined as a measurable classification component instead of a purely black-box agent.


## Why this matters

In a financial setting, correctness and safety matter as much as functionality.

This version emphasizes:
- reliability in routing behavior
- safety in SQL execution
- interpretability of both successes and failures

The goal is to support financial decision-related queries in a way that is easier to analyze and verify.


## Relation to later versions

- `router-sql-evaluation`: midterm-stage Router + SQL development with evaluation focus
- `router-sql`: later post-reorg improvements, mainly robustness and edge-case handling

This branch represents the earlier analytical and evaluation-oriented version of the Router + SQL module.
