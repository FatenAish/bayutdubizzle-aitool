import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

st.markdown(
    """
    <h1 style="font-size:32px; font-weight:800; text-align:center; margin-bottom:5px;">
        <span style="color:#0E8A6D;">Bayut</span> & 
        <span style="color:#D71920;">Dubizzle</span> AI Content Assistant
    </h1>
    <p style="text-align:center; color:#4b5563; font-size:14px; margin-bottom:25px;">
        Fast internal knowledge search powered by internal content (.txt files in /data).
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# PATHS
# -------------------------------
DATA_DIR = "data"
FAISS_DIR = os.path.join(DATA_DIR, "faiss_store")

st.write(f"📁 Using data folder: `{DATA_DIR}`")

# -------------------------------
# EMBEDDINGS & LLM
# -------------------------------
@st.cache_resource
def get_embeddings():
    # Uses OpenAI Embeddings – light and no torch
    return OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource
def get_llm():
    # Make sure OPENAI_API_KEY is set in Railway variables
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

# -------------------------------
# BUILD / LOAD VECTORSTORE
# -------------------------------
def build_vectorstore():
    """Index all .txt files in /data."""
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        return None

    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    )

    for fname in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fname)
        if os.path.isfile(path) and fname.lower().endswith(".txt"):
            try:
                loader = TextLoader(path, encoding="utf-8")
                file_docs = loader.load()
                docs.extend(splitter.split_documents(file_docs))
            except Exception:
                continue

    if not docs:
        return None

    vs = FAISS.from_documents(docs, get_embeddings())
    vs.save_local(FAISS_DIR)
    return vs

@st.cache_resource
def load_or_build_vectorstore():
    """Load FAISS index if exists; otherwise build a new one."""
    if os.path.isdir(FAISS_DIR):
        try:
            return FAISS.load_local(
                FAISS_DIR,
                get_embeddings(),
                allow_dangerous_deserialization=True
            )
        except Exception:
            pass
    return build_vectorstore()

# -------------------------------
# INIT VECTORSTORE
# -------------------------------
vectorstore = load_or_build_vectorstore()
if vectorstore is None:
    st.warning("No index built yet. Add .txt files to /data and click 'Rebuild Index'.")
else:
    st.success("Vector index is ready ✅")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

# -------------------------------
# CHAT HISTORY
# -------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------------------
# INPUT
# -------------------------------
st.subheader("Ask your internal question")

col_q, col_btn = st.columns([4, 1])
with col_q:
    user_query = st.text_input("Question", key="question_input")
with col_btn:
    ask_clicked = st.button("Ask")

# -------------------------------
# HANDLE QUESTION
# -------------------------------
if ask_clicked and user_query.strip():
    if retriever is None:
        st.error("Index not ready. Add .txt files to /data and rebuild the index.")
    else:
        with st.spinner("Thinking..."):
            # Get relevant docs
            docs = retriever.invoke(user_query)
            context = "\n\n".join(d.page_content for d in docs)

            prompt = PromptTemplate.from_template(
                "Use ONLY the context below to answer the question.\n"
                "If you don't find the answer, say you don't know based on internal docs.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer in one clear paragraph:"
            )

            chain = prompt | get_llm() | StrOutputParser()
            answer = chain.invoke({"context": context, "question": user_query})

        # Save to history
        st.session_state["history"].append(
            {"q": user_query, "a": answer, "sources": docs}
        )

# -------------------------------
# SHOW CHAT HISTORY (oldest at bottom)
# -------------------------------
st.markdown("---")
st.markdown("### Conversation")

# Show from oldest to newest
for item in st.session_state["history"]:
    st.markdown(f"**❓ Question:** {item['q']}")
    st.markdown(f"**✅ Answer:** {item['a']}")
    if item["sources"]:
        with st.expander("📎 Evidence from documents"):
            for i, d in enumerate(item["sources"], 1):
                st.markdown(f"**{i}. Source:** {d.metadata.get('source', 'Unknown')}")
                st.write(d.page_content[:500] + "...")
    st.markdown("---")

# -------------------------------
# REBUILD INDEX BUTTON
# -------------------------------
if st.button("🔄 Rebuild Index"):
    with st.spinner("Rebuilding vector index from /data..."):
        # Clear cache for vectorstore
        load_or_build_vectorstore.clear()
        vs_new = load_or_build_vectorstore()
    if vs_new is None:
        st.error("No .txt documents found in /data. Add files and try again.")
    else:
        st.success("Index rebuilt successfully! ✅")
