import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime
import os

from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

load_dotenv(PROJECT_ROOT / ".env", override=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_interface import chat


def _resolve_from_root(p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _tail_text(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _summarize_with_llamaindex(text: str, model: str) -> str:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.llms.openai import OpenAI as LIOpenAI

    llm = LIOpenAI(model=model, temperature=0.0)
    docs = [Document(text=text, metadata={"source": "sql_results"})]
    index = VectorStoreIndex.from_documents(docs)
    qe = index.as_query_engine(llm=llm)

    questions = [
        "How many total cases are in this log? How many have ERROR?",
        "List each ERROR case as: [case_number] query -> error message.",
        "How many cases indicate fallback (keywords: fallback / unsafe / LLM failed)? List those case numbers.",
        "Summarize overall quality in 3 bullet points and give 3 actionable fixes for the SQL agent.",
    ]

    out = []
    for q in questions:
        ans = qe.query(q)
        out.append(f"Q: {q}\nA: {ans}\n")
    return "\n".join(out)


def _summarize_with_openai(text: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    prompt = f"""
You are reviewing an evaluation log from a Banking Agent SQL module.

Return:
1) total cases + ERROR count
2) list ERROR cases as: [case] query -> error
3) fallback count (unsafe SQL / LLM failed) + case numbers
4) 3 bullet overall assessment
5) 3 concrete fixes

Log:
{text}
""".strip()

    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--customer-id", default="12345")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--summary-model", default=os.getenv("SQL_SUMMARY_MODEL", "gpt-4.1-mini"))
    parser.add_argument(
        "--summary-tail-lines",
        type=int,
        default=int(os.getenv("SQL_SUMMARY_TAIL_LINES", "800")),
    )

    args = parser.parse_args()

    csv_path = _resolve_from_root(args.csv)
    customer_id = args.customer_id

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cases = [row["text"].strip() for row in reader if row.get("text", "").strip()]

    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sql_results_{ts}.txt"

    def vprint(msg: str):
        if args.verbose:
            print(msg, flush=True)

    vprint(f"[INFO] Total cases: {len(cases)}")
    vprint(f"[INFO] Writing to: {out_path}")

    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"Total cases: {len(cases)}\n\n")
        for i, q in enumerate(cases, start=1):
            out.write("=" * 80 + "\n")
            out.write(f"[{i:02d}] QUERY: {q}\n")

            vprint("=" * 80)
            vprint(f"[{i:02d}/{len(cases):02d}] QUERY: {q}")

            try:
                result = chat(
                    user_input=q,
                    customer_id_number=customer_id,
                    session_id=f"sql-case-{i}",
                    summary=False,
                )
                ans = str(result.get("final_answer", ""))

                out.write("FINAL ANSWER:\n")
                out.write(ans + "\n")

                vprint("[OK] Answer:")
                vprint(ans)

            except Exception as e:
                out.write(f"ERROR: {repr(e)}\n")
                vprint(f"[ERROR] {repr(e)}")

    print(f"Done. Wrote: {out_path}")

    if args.summary:
        log_text = out_path.read_text(encoding="utf-8", errors="replace")
        log_text = _tail_text(log_text, args.summary_tail_lines)
        model = args.summary_model

        print("\n" + "=" * 80)
        print("LlamaIndex Summary (auto)")
        print("=" * 80)

        try:
            summary = _summarize_with_llamaindex(log_text, model=model)
        except Exception as e:
            print(f"[WARN] LlamaIndex summary failed: {type(e).__name__}: {e}")
            summary = _summarize_with_openai(log_text, model=model)

        print(summary)

        summary_path = out_dir / f"sql_summary_{ts}.txt"
        summary_path.write_text(summary, encoding="utf-8")
        print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()