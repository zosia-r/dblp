# dblp

DBLP analytics and exploration project with ETL, EDA, topic discovery, and RAG pipelines.

## Dashboard
The dashboard is built with Streamlit and provides an interactive interface for exploring the DBLP dataset, visualizing topics, and performing RAG-based question answering.
It is deployed on Streamlit Cloud and can be accessed at: https://zosia-r-dblp--dblp-project-4qih6i.streamlit.app/

## Project structure

```text
├── 📚_DBLP_Project.py
├── main.py
├── pages/
│   ├── 1_⚙️_ETL.py
│   ├── 2_🗂️_Topic_Discovery.py
│   ├── 3_📊_EDA.py
│   └── 4_👽_RAG.py
├── results/
├── sample_data/
└── src/
    ├── eda/
    ├── etl/
    ├── hf/
    ├── rag/
    └── topic_modeling/
```

## What each part does

- `main.py`: command-line interface for preparing data.
- `📚_DBLP_Project.py`, `main.py`: dashboard entry points.
- `pages/`: Streamlit multi-page UI (ETL, topic discovery, EDA, RAG).
- `results/`: generated summary metrics and analysis outputs.
- `sample_data/`: lightweight demo dataset.
- `src/etl/`: extraction, parsing, transformation, validation pipeline.
- `src/eda/`: exploratory analysis logic and topic-focused EDA helpers.
- `src/topic_modeling/`: BERTopic configuration, training, persistence, stats.
- `src/rag/`: retriever-generator pipeline and data loading for QA/search.
- `src/hf/`: Hugging Face model/data load-upload utilities.

## Running the project
To prepare the data, run:
```bash
uv run main.py <path_to_dblp_xml>
```

<!-- ## GPU / RAPIDS setup

`uv add rapids` is not the same thing as installing the RAPIDS GPU stack for BERTopic.

For the GPU path, use a separate Conda or Mamba environment with a CUDA-matched RAPIDS build. The current repo environment is still fine for the CPU path, but GPU clustering needs cuML/cuDF/cuPy from RAPIDS.

Example for CUDA 12 on Linux:

```bash
mamba create -n dblp-rapids python=3.12 -y
mamba activate dblp-rapids
mamba install -c rapidsai -c conda-forge -c nvidia \
    cuml-cuda12x cudf-cuda12x cupy-cuda12x -y
```

Then run the project from that environment. If `cuml` is importable, the topic-modeling code will use GPU UMAP/HDBSCAN automatically; otherwise it falls back to the CPU implementation. -->
