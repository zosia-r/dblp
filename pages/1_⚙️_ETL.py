import streamlit as st


st.title("⚙️ ETL - Extract, Transform, Load")

# --- Helpers ---
@st.cache_data
def build_er_diagram():
    return """
    digraph ER {
        rankdir=LR;

        papers [shape=box];
        authors [shape=box];
        author_aliases [shape=box];
        paper_authors [shape=box];

        papers -> paper_authors;
        authors -> paper_authors;
        authors -> author_aliases;
    }
    """

st.set_page_config(page_title="ETL",
                   page_icon="⚙️",
                    layout="wide")

st.markdown("""
This page describes the ETL pipeline implemented to process the raw DBLP XML dataset
             and load it into a structured SQLite database for analysis.
""")

with st.expander("Pipeline Description", expanded=False):

    st.markdown("""
    ### Model pipeline (ETL system)

    | Phase | Component | Description |
    |------|----------|-------------|
    | **Phase 0** | Dataset profiling | Basic statistics, schema inspection, row counts |
    | **Phase 1** | XML streaming (`lxml`) | Incremental parsing of large DBLP XML dump |
    | **Phase 2** | Interim storage | CSV files for intermediate representation |
    | **Phase 3** | Author resolution | Author identity matching and paper assignment |
    | **Phase 4** | SQLite loading | Insert into `dblp.db` |
    | **Phase 5** | Validation | Row counts +  integrity checks |

    Each phase is implemented as a standalone script with clear inputs/outputs, allowing for modular execution and easy debugging.
    """)

st.divider()


st.markdown("""
## Data Distribution
            
An initial scan of the **DBLP - Computer Science Bibliography dataset** showed the following distribution of record types:

###### Total records: 12501682
            
- **Article:** 4249812
- **Inproceeding:** 3860628
- **Proceeding:** 63924
- **Book:** 21374
- **Incollection:** 71063
- **Phdthesis:** 152212
- **Mastersthesis:** 27
- **Www:** 4059905
- **Person:** 0
- **Data:** 22737   
            
---

## Data Selection
            
Considering this distribution, I decided the analysis should focus on two main types of publications:
- **article** - An article from a journal or magazine - 34% of records
- **inproceedings** - A paper in a conference or workshop proceedings - 31% of records
            
These types were selected because:

- they represent the majority of records  
- they share a consistent schema  
- they are the most relevant for analyzing research trends in computer science  
            
Other record types were excluded due to lower consistency and limited analytical value.
            
In addition, I decieded to focus only on records from years **2010-2025**, as they represent the most recent and relevant data for trend analysis.

---

## Author Data
            
In addition, the `<www>` records were used to conduct author-level analysis. These records make it possible to:
- extract author names and their variants (aliases)
- build links between authors and publications  
- better distinguish between authors with similar names  

---

## Extracted Fields

For each **publication** (article or inproceeding), the following fields were extracted:

- **paper_id** (unique identifier)
- **title**
- **year**
- **venue** (journal or conference)
- **type** (article / inproceedings)
            
Records with missing core fields were excluded to maintain data quality.

**Author** data is stored as:

- **author_id** (unique identifier)
- **primary_name** (most frequent name variant)
- **alias** (all observed name variants)

---
            
## Database Schema
""")

dot = """
digraph ER {
    rankdir=LR;

    papers [shape=box];
    authors [shape=box];
    author_aliases [shape=box];
    paper_authors [shape=box];

    papers -> paper_authors;
    authors -> paper_authors;
    authors -> author_aliases;
}
"""

dot = build_er_diagram()
st.graphviz_chart(dot)

st.markdown("""
---
## Dataset Summary after ETL

| Metric | Value |
|--------|------|
| Total papers | 5 927 455 |
| Articles | 3 245 977 |
| Inproceedings | 2 681 478 |
| Author identities | 4 057 541 |
| Author aliases | 111 838 |
| Paper authors | 22 533 652 |

""")
