import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from utils import process_document, list_pdf_files
from concurrent.futures import ThreadPoolExecutor, as_completed

INDEX_NAME = "bank-docs-index"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
NAMESPACE = "deposits-en"

# initialize OpenAI and Pinecone clients, create index if not exists
def init_clients():
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY does not exist")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY does not exist")

    # initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    # initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)

    # create / get index
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        print(f"Index '{INDEX_NAME}' does not exist, creating...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" #
            ),
        )
        print(f"Index '{INDEX_NAME}' has been created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' exists, using existing index.")

    index = pc.Index(INDEX_NAME)
    return client, index


def process_folder(folder_path: str, namespace: str = NAMESPACE, max_chunks: int = None):
    client, index = init_clients()

    pdf_files = list_pdf_files(folder_path)
    if not pdf_files:
        print(f"no pdf files found in folder: {folder_path}")
        return

    success_count, total_count = 0, len(pdf_files)
    max_workers = min(8, total_count)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        temp = {}
        for idx, pdf_path in enumerate(pdf_files, start=1):
            future = executor.submit(
                process_document,
                pdf_path,
                index,
                namespace,
                client,
            )
            temp[future] = (idx, pdf_path)

        for future in as_completed(temp):
            idx, pdf_path = temp[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{idx}/{total_count}] file {pdf_path} processing failed: {e}")
            else:
                print(f"[{idx}/{total_count}] file {pdf_path} processing completed successfully.")
                success_count += 1
    print(f"all files processed. successfully: {success_count}/{total_count}")


if __name__ == "__main__":
    FOLDER_PATH = "/Users/zhumiban/Desktop/agent_bank/bank_docs"

    process_folder(
        folder_path=FOLDER_PATH,
        namespace=NAMESPACE,
    )
