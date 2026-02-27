import csv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from chat_interface import chat

def main():
    csv_path = "sql_30_cases.csv"
    customer_id = "12345"
    out_path = "sql_30_results.txt"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cases = [row["text"].strip() for row in reader if row.get("text", "").strip()]

    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"Total cases: {len(cases)}\n\n")
        for i, q in enumerate(cases, start=1):
            out.write("=" * 80 + "\n")
            out.write(f"[{i:02d}] QUERY: {q}\n")
            try:
                result = chat(user_input=q, customer_id_number=customer_id, session_id=f"sql-case-{i}", summary=False)
                out.write("FINAL ANSWER:\n")
                out.write(result.get("final_answer", "") + "\n")
            except Exception as e:
                out.write(f"ERROR: {repr(e)}\n")

    print(f"Done. Wrote: {out_path}")

if __name__ == "__main__":
    main()
