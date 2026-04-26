import streamlit as st

st.set_page_config(page_title="DBLP Project", page_icon="📚")

st.markdown("""
# DBLP Data Engineering Project

### *Zofia Różańska*

---

### Project Scope

1. **ETL pipeline** - extracting data from the DBLP XML dataset, transforming it into a structured format, and loading it into a SQL database.
2. **Exploratory Data Analysis** - performing exploratory data analysis (EDA) to uncover trends and insights in the publication data.
3. **Topic modeling** - applying natural language processing techniques to identify research topics and trends over time.  
4. **RAG chatbot** - building a Retrieval-Augmented Generation (RAG) chatbot that can answer questions about the DBLP dataset using the structured data and insights obtained from the analysis.


""")