import streamlit as st
import pandas as pd

st.title("📊 EDA")


st.markdown("""
# DBLP Data Analysis Dashboard

## Overview

This dashboard presents an analysis of the DBLP dataset — a large collection of computer science publications.

Because the full dataset is very large, a curated subset was selected to ensure efficient processing and consistent structure.

The goal of this project is not only to analyze the data, but also to design a clear and scalable data pipeline.

---

## Data Selection
            
An initial scan of the DBLP dataset showed the following distribution of record types:

| Element | Count |
|--------|------|
| Total papers | 12501682 |
| Article | 4249812 |
| Inproceeding | 3860628 |
| Proceeding | 63924 |
| Book | 21374 |
| Incollection | 71063 |
| Phdthesis | 152212 |
| Mastersthesis | 27 |
| Www | 4059905 |
| Person | 0 |
| Data | 22737 |   
            
Based on this distribution, the analysis focuses on two main types of publications:

- **article** (journal papers)
- **inproceedings** (conference papers)
            
These types were selected because:

- they represent the majority of structured records  
- they share a consistent schema  
- they are the most relevant for analyzing research trends in computer science  
            
Other record types (such as books, theses, or web entries) were excluded due to lower consistency and limited analytical value.


In addition, `<www>` records were used to improve author-level analysis. These records make it possible to:

- build links between authors and publications  
- analyze collaboration networks  
- group different name variants under the same author  
- better distinguish between authors with similar names  

---

## Extracted Fields

For each publication (article or inproceedings), the following fields were extracted:

- **paper_id** (unique identifier)
- **title**
- **year**
- **venue** (journal or conference)
- **type** (article / inproceedings)
            
Records with missing core fields were excluded to maintain data quality.

Author data is stored as:

- **author_id**
- **primary_name** (most frequent name variant)
- **alias** (all observed name variants)

This approach allows basic author grouping without using complex disambiguation methods.

---

## Dataset Summary

| Metric | Value |
|--------|------|
| Total papers | 5927455 |
| Articles | 3245977 |
| Inproceedings | 2681478 |
| Author identities | 4057541 |
| Author mentions | 22534080 |

---

## Technologies Used

This project follows a simple and modular ETL pipeline:

**XML → CSV → SQL**

Main components:

- **Python** – core language for data processing  
- **ElementTree (iterparse)** – streaming XML parsing  
- **Pandas** – data transformation and CSV handling  
- **SQLite** – relational storage and querying  
- **Streamlit** – interactive dashboard  

The pipeline is designed to handle large files efficiently by processing data in a streaming manner and using CSV as an intermediate layer.

---

## Notes

- Author names are normalized using simple rules (lowercasing, trimming, removing punctuation)  
- No advanced entity resolution or ML-based matching was applied  
- This reduces the risk of incorrect merges and keeps the pipeline deterministic  

---

This project focuses on building a clean data pipeline and enabling reliable exploratory analysis.
""")