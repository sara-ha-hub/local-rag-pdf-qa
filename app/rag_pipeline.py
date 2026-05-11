"""
RAG Pipeline Core
-----------------
Handles PDF loading, chunking, embedding, vector store, and retrieval.
Uses only free/local models via HuggingFace + FAISS.
"""

from operator import itemgetter
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# ── Constants ────────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"   # small, CPU-friendly, instruction-tuned
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50
TOP_K       = 4

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


# ── PDF Loading & Chunking ────────────────────────────────────────────────────
def load_and_split_pdf(pdf_path: str) -> List:
    """Load a PDF and split into overlapping chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ── Embeddings (shared helper to avoid duplication) ──────────────────────────
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Vector Store ─────────────────────────────────────────────────────────────
def build_vectorstore(chunks: List) -> FAISS:
    """Embed chunks and build a FAISS vector store."""
    return FAISS.from_documents(chunks, _get_embeddings())


def save_vectorstore(vectorstore: FAISS, path: str = "vectorstore") -> None:
    vectorstore.save_local(path)


def load_vectorstore(path: str = "vectorstore") -> FAISS:
    return FAISS.load_local(
        path, _get_embeddings(), allow_dangerous_deserialization=True
    )


# ── LLM ──────────────────────────────────────────────────────────────────────
def load_llm() -> HuggingFacePipeline:
    """Load a lightweight causal LM locally for text-generation."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,         # deterministic for QA
        return_full_text=False,  # only the generated continuation
        device=-1,               # CPU
    )
    return HuggingFacePipeline(pipeline=pipe)


# ── RAG Chain ─────────────────────────────────────────────────────────────────
def build_rag_chain(vectorstore: FAISS, llm: HuggingFacePipeline):
    """Build the full RAG chain with retriever + LLM (LCEL)."""
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    def format_docs(docs) -> str:
        return "\n\n".join(d.page_content for d in docs)

    # Step 1: fetch query + retrieved source docs in parallel
    retrieve = RunnableParallel(
        query=itemgetter("query"),
        source_documents=itemgetter("query") | retriever,
    )

    # Step 2: build the answer from the retrieved documents
    answer_chain = (
        {
            "context": RunnableLambda(lambda x: format_docs(x["source_documents"])),
            "question": itemgetter("query"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step 3: combine — keep query + source_documents, add `result`
    chain = retrieve | RunnablePassthrough.assign(result=answer_chain)
    return chain


# ── Main Entry ────────────────────────────────────────────────────────────────
def process_pdf_and_query(pdf_path: str, query: str) -> Tuple[str, List]:
    """
    Full pipeline: load PDF → chunk → embed → retrieve → answer.
    Returns (answer, source_chunks).
    """
    print("[1/4] Loading and splitting PDF...")
    chunks = load_and_split_pdf(pdf_path)
    print(f"      → {len(chunks)} chunks created")

    print("[2/4] Building vector store...")
    vectorstore = build_vectorstore(chunks)

    print(f"[3/4] Loading LLM ({LLM_MODEL})...")
    llm = load_llm()

    print("[4/4] Running RAG chain...")
    chain = build_rag_chain(vectorstore, llm)
    result = chain.invoke({"query": query})

    return result["result"], result["source_documents"]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python rag_pipeline.py <pdf_path> <question>")
        sys.exit(1)
    answer, sources = process_pdf_and_query(sys.argv[1], sys.argv[2])
    print("\n=== Answer ===")
    print(answer)
    print(f"\n=== {len(sources)} source chunks used ===")
