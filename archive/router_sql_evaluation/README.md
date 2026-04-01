# Archive: Router + SQL (Midterm Version)

This folder contains the Router and SQL work from the midterm stage, before the system was reorganized.

## Overview

This version focuses on making the Router and SQL components more reliable and analyzable in a financial setting.

## What this module does

- Classifies user queries into predefined intent types
- Routes transaction-related queries to the SQL component
- Retrieves structured financial data from the database

## Key design ideas

### Routing as classification
- Treats routing as an intent classification problem
- Produces structured and consistent outputs
- Makes errors easier to analyze

### Evaluation and reliability
- Includes evaluation using:
  - accuracy
  - precision / recall / F1
  - confusion matrix
- Used to analyze misclassification and ambiguous queries

### SQL safety
- Enforces read-only (SELECT only)
- Applies LIMIT constraints
- Filters unsafe queries before execution

## Components

### Router
- Converts user input into intent labels
- Handles malformed outputs safely

### SQL Agent
- Generates SQL from natural language
- Applies safety checks
- Includes fallback when generation fails

## Notes

This version is kept for reference to show:
- the evaluation setup
- the design choices around reliability and safety

Later work (in `router-sql` and `reorg`) focuses more on robustness and integration.
