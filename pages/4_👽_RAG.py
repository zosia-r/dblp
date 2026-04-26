import streamlit as st
from src.rag.pipeline import RAGPipeline

st.set_page_config(page_title="DBLP RAG Chatbot", 
                   page_icon="👽",
                   layout="wide")

st.title("👽 DBLP RAG Chatbot")

# @st.cache_resource
# def load_pipeline():
#     return RAGPipeline()

# pipeline = load_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Zadaj pytanie o publikacje:")

# if query:
#     answer, docs = pipeline.run(query)
#     st.session_state.history.append((query, answer, docs))

# for q, a, docs in reversed(st.session_state.history):
#     st.markdown(f"### 🧑 {q}")
#     st.markdown(f"**🤖 {a}**")

#     with st.expander("🔎 Źródła"):
#         for d in docs:
#             st.write("-", d)