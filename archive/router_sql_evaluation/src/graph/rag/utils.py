from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import uuid
import os

# PDFLoader class to extract text from a PDF file
class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

# fetch all pdf files in a given folder
def list_pdf_files(folder_path: str):
    files = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if os.path.isfile(full_path) and name.lower().endswith(".pdf"):
            files.append(full_path)
    return files

# split text using LangChain's RecursiveCharacterTextSplitter
def split_text_with_langchain(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。", "!", "?", ".", "!", "?",
            ",", ",", " ", ""
        ],
    )
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]

# embed text using OpenAI's Embeddings API
def get_embeddings(texts, client, model="text-embedding-3-small"):
    single_input = False
    if isinstance(texts, str):
        texts = [texts]
        single_input = True

    response = client.embeddings.create(
        model=model,
        input=texts
    )
    vectors = [item.embedding for item in response.data]
    return vectors[0] if single_input else vectors


# build pinecone vectors
def build_pinecone_vectors(chunks, source, client):
    # generate embeddings
    embeddings = get_embeddings(chunks, client=client)

    # build vectors with metadata
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vec_id = str(uuid.uuid4())
        metadata = {
            "text": chunk,
            "chunk_index": i,
            "source": source,
        }
        vectors.append(
            {
                "id": vec_id,
                "values": emb,
                "metadata": metadata
            }
        )
    return vectors

# upsert vectors to pinecone
def upsert_vectors(index, vectors, namespace):
    index.upsert(
        vectors=vectors,
        namespace=namespace
    )

# split and insert document into pinecone
def process_document(pdf_path, index, namespace="default", client=None):
    # 1. load pdf and extract text
    loader = PDFLoader(pdf_path)
    full_text = loader.extract_text()

    # 2. split text using LangChain's RecursiveCharacterTextSplitter
    chunks = split_text_with_langchain(
        full_text,
        chunk_size=1000,
        chunk_overlap=200 # industrial golden standard, can change if PRD needs
    )
    print(f"Total # of {len(chunks)} chunks were generated.")

    # 3. build pinecone vectors
    vectors = build_pinecone_vectors(
        chunks,
        source=pdf_path,
        client=client
    )
    # 4. upsert vectors to pinecone
    upsert_vectors(index, vectors, namespace)
    print(f"Successfully inserted {len(chunks)} chunks to Pinecone.")

# query pinecone to get relevant chunks
def query_pinecone(index, query_vector, namespace, top_k=10, include_metadata=True, filter=None):
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=include_metadata,
        namespace=namespace,
        filter=filter,
    )
    return response

# convert pinecone query response to documents for rerank in a format that Cohere rerank expects:
# {"id": <id>, "chunk_text": <text>}
def build_rerank_documents(query_response, text_field="text"):
    matches = query_response.get("matches", [])
    documents = []

    for matched in matches:
        metadata = matched.get("metadata", {}) or {}
        text = metadata.get(text_field)
        if not text:
            continue

        documents.append({
            "id": matched.get("id"),
            "text": text,
            "original_score": matched.get("score")
        })
    return documents

# rerank documents using Cohere rerank model
def rerank_documents(pc, query, documents, top_n=5, model="bge-reranker-v2-m3", \
                     text_field="text", extra_parameters=None, return_documents=True):
    if not documents:
        return {"results": []}

    params = {"truncate": "END"}
    if extra_parameters:
        params.update(extra_parameters)

    ranked_results = pc.inference.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        rank_fields=[text_field],
        return_documents=return_documents,
        parameters=params
    )
    return ranked_results

# search with rerank full pipeline:
# 1. vector query
# 2. rerank
# 3. return reranked results
def search_with_rerank(pc, index, query, query_vector, namespace, vector_top_k=10, rerank_top_n=5, \
                       text_field="text", filter=None):
    # 1. vector query
    initial_response = query_pinecone(
        index=index,
        query_vector=query_vector,
        namespace=namespace,
        top_k=vector_top_k,
        include_metadata=True,
        filter=filter
    )

    # 2. convert to documents
    docs = build_rerank_documents(initial_response, text_field=text_field)

    # 3. rerank documents
    reranked = rerank_documents(
        pc=pc,
        query=query,
        documents=docs,
        top_n=rerank_top_n,
        text_field=text_field,
    )
    return reranked