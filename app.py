# ============================================================
# app.py – Dubizzle Group AI Content Lab
# FULL WORKING VERSION (FASTEST + CHAT + EVIDENCE + RAG)
# ============================================================

import os
import shutil
import pandas as pd
import streamlit as st

# ==============================================
# PAGE CONFIG
# ==============================================
st.set_page_config(page_title="Dubizzle Group AI Content Lab", layout="wide")

# ==============================================
# GLOBAL CLEAN CSS
# ==============================================
st.markdown("""
<style>
[data-testid="stVerticalBlock"] > div {
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
main .block-container { padding-top: 0rem !important; }
.header-wrapper { text-align: center !important; width: 100%; }
.header-title { font-size: 32px; font-weight: 900; color: #111827; }
.header-title .red { color: #D92C27 !important; }
.header-sub { font-size: 15px; color: #4b5563; margin-bottom: 20px !important; }

.bubble { 
  padding: 12px 16px; 
  border-radius: 14px; 
  margin: 6px 0; 
  max-width: 85%; 
  line-height: 1.5;
  font-size: 15px;
}
.bubble.user { background: #f2f2f2; margin-left: auto; }
.bubble.ai { background: #ffffff; margin-right: auto; }

.evidence { 
  background:#fafafa; 
  border:1px solid #e5e7eb; 
  border-radius:12px; 
  padding:10px; 
  margin:10px 0; 
}
</style>
""", unsafe_allow_html=True)

# ==============================================
# HEADER
# ==============================================
st.markdown('<div class="header-wrapper">', unsafe_allow_html=True)
st.markdown('<div class="header-title"><span class="red">Dubizzle Group</span> AI Content Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Internal AI-powered content platform for Bayut & Dubizzle teams</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==============================================
# SIDEBAR TOOL SELECTOR
# ==============================================
tool = st.sidebar.selectbox(
    "Choose a tool",
    [
        "LPV Rewriter",
        "Guidelines Checker",
        "Arabic Grammar Fixer",
        "English Grammar Fixer",
        "Content Brief Generator",
        "Trend Analyzer",
        "Sheet Analyzer",
        "Internal RAG",
    ],
)

# Placeholder Tools
if tool != "Internal RAG":
    st.subheader(f"{tool}")
    st.write("🚧 Coming soon")
    st.stop()

# =====================================================================
# ✅✅✅ INTERNAL RAG (FASTEST, CLEANEST, CHATGPT-STYLE CHAT)
# =====================================================================

st.subheader("Internal Knowledge Base (Local RAG)")
st.caption("FAST RAG • Uses documents from /data • Delete /data/faiss_store to rebuild")

# ✅ Correct imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ====== CONFIG ======
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")

# ✅ Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Working Groq LLM
def get_local_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

# ====== LOAD DOCUMENTS ======
def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf": return PyPDFLoader(path).load()
        if ext == ".docx": return Docx2txtLoader(path).load()
        if ext in [".txt", ".md"]: return TextLoader(path).load()
        if ext == ".csv": return CSVLoader(path, encoding="utf-8").load()
        if ext == ".xlsx":
            df = pd.read_excel(path)
            return [{"page_content": df.to_string(), "metadata": {"source": path}}]
    except:
        print("Skipping bad file:", path)
    return []

def load_default_docs():
    docs = []
    if not os.path.isdir(DATA_DIR): return docs
    for f in os.listdir(DATA_DIR):
        if f == "faiss_store": continue
        p = os.path.join(DATA_DIR, f)
        if os.path.isfile(p):
            docs.extend(load_document(p))
    return docs

def faiss_exists():
    return os.path.isdir(INDEX_DIR)

def save_faiss(store):
    os.makedirs(DATA_DIR, exist_ok=True)
    store.save_local(INDEX_DIR)

def load_faiss():
    return FAISS.load_local(
        INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    return FAISS.from_texts(texts, get_embeddings())

# ===== BUTTONS =====
c1, c2 = st.columns([1, 1])

if c1.button("🔄 Rebuild Index"):
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    st.cache_resource.clear()
    st.success("Index cleared. Restart app to rebuild.")
    st.stop()

if c2.button("🧹 Clear Chat"):
    st.session_state.pop("rag_history", None)
    st.rerun()

# ===== LOAD OR CREATE INDEX =====
if faiss_exists():
    with st.spinner("Loading FAISS index..."):
        vectorstore = load_faiss()
    st.success("✅ Index loaded")
else:
    docs = load_default_docs()
    if not docs:
        st.error("❌ No documents in /data")
        st.stop()

    with st.spinner("Indexing documents..."):
        vectorstore = build_vectorstore(docs)
        save_faiss(vectorstore)

    st.success("✅ Index created")

# ===== CHAT HISTORY =====
if "rag_history" not in st.session_state:
    st.session_state["rag_history"] = []

for q, a in st.session_state["rag_history"]:
    st.markdown(f"<div class='bubble user'>{q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{a}</div>", unsafe_allow_html=True)

# ===== ASK A QUESTION =====
query = st.text_input("Ask your question:")

if query:

    with st.spinner("Thinking..."):

        # Evidence preview
        hits = vectorstore.similarity_search_with_score(str(query), k=3)

        # ✅ Stable retriever
        rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # ✅ Stage 1 — Extract only the text from dict input
        extract_question = RunnableLambda(lambda x: x["question"])

        # ✅ Stage 2 — Pass the string to retriever
        retrieve_docs = extract_question | rag_retriever

        # ✅ Stage 3 — Process retrieved docs
        prepare_context = RunnableLambda(
            lambda docs: "\n\n".join(d.page_content[:1500] for d in docs)
        )

        # ✅ Combine retrieval pipeline
        context_pipeline = retrieve_docs | prepare_context

        # ✅ Prompt
        prompt = PromptTemplate.from_template(
            "Use ONLY this context to answer:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        # ✅ FINAL CHAIN
        chain = (
            {
                "context": context_pipeline,
                "question": RunnablePassthrough(),
            }
            | prompt
            | get_local_llm()
            | StrOutputParser()
        )

        answer = chain.invoke({"question": query})

    # ✅ UI — chat history
    st.session_state.setdefault("rag_history", []).append((query, answer))
    st.markdown(f"<div class='bubble user'>{query}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{answer}</div>", unsafe_allow_html=True)

    # ✅ Evidence output
    if hits:
        st.markdown("### 📎 Evidence")
        for i, (doc, score) in enumerate(hits, 1):
            snippet = doc.page_content[:400]
            st.markdown(
                f"<div class='evidence'><b>{i}.</b> similarity={score:.3f}<br>{snippet}...</div>",
                unsafe_allow_html=True
            )
