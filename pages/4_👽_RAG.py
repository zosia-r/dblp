import streamlit as st
from src.rag.pipeline import RAG

st.set_page_config(page_title="DBLP RAG Chatbot", 
                   page_icon="👽",
                   layout="wide")

st.title("👽 DBLP RAG Chatbot")

@st.cache_resource
def load():
    return RAG()

rag = load()

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question about publications:")

if query:
    answer, docs = rag.run(query)
    st.session_state.history.append((query, answer, docs))

for q, a, docs in reversed(st.session_state.history):
    st.markdown(f"### 🧑 {q}")
    st.markdown(f"**🤖 {a}**")

    with st.expander("🔎 Sources"):
        for d in docs:
            st.write("-", d)