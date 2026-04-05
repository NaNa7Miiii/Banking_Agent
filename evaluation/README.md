## Evaluation

We evaluate both the routing (planner) and SQL components.

### Router Evaluation

- Accuracy: 0.8293
- Macro F1: 0.8185

The routing module correctly assigns ~83% of queries to the appropriate sub-agent, with balanced performance across task categories.

### SQL Evaluation

- SQL routing rate: 1.00
- Execution success rate: 1.00
- Non-empty answer rate: 1.00
- Error rate: 0.00

The SQL agent demonstrates strong reliability in query generation and execution.

### Key Insight

Most errors arise from **semantic ambiguity**, especially between fraud and transaction queries, rather than system design flaws.
