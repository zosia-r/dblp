import sys
from pathlib import Path

import streamlit as st

# ── resolve src/eda regardless of where streamlit is launched from ────────────
_ROOT = Path(__file__).resolve().parent.parent  # project root (above pages/)
sys.path.insert(0, str(_ROOT / "src" / "eda"))

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DBLP EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── imports from EDA modules ─────────────────────────────────────────────────
from src.eda.eda_general import (
    load_data,
    table_overview,
    table_null_detail,
    table_papers_summary,
    table_top_venues,
    table_top_authors,
    publications_per_year,
    publications_per_year_by_type,
    type_distribution_pie,
    cumulative_publications,
    yoy_growth_rate,
    top_venues_bar,
    top_conferences_vs_journals,
    venue_heatmap,
    venue_bump_chart,
    venue_treemap,
    venue_sankey,
    unique_venues_per_year,
    authors_per_paper_dist,
    authors_per_paper_by_type,
    median_authors_per_year,
    top_authors_bar,
    author_productivity_dist,
    author_activity_span,
    new_authors_per_year,
    author_retention_funnel,
    author_scatter_papers_vs_years,
    author_parallel_coordinates,
    coauthorship_weight_dist,
    top_authors_by_degree,
    avg_collaborators_per_year,
    title_length_dist,
    density_heatmap_year_vs_authors,
    sunburst_year_type_venue,
)
from src.eda.eda_topics import (
    load_topics,
    table_topics_overview,
    table_topic_type_breakdown,
    table_top_authors_per_topic,
    topics_papers_bar,
    topics_pie,
    topics_treemap,
    topics_over_time_line,
    topics_share_over_time,
    topic_heatmap_year,
    topic_bump_chart,
    topic_yoy_growth,
    emerging_topics,
    venue_topic_heatmap,
    topic_venue_sunburst,
    top_authors_per_topic_bar,
    author_topic_diversity,
    author_specialisation_scatter,
    topic_cooccurrence_heatmap,
    topic_type_stacked_bar,
    null_topic_over_time,
)

# ── data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Parquet files…")
def get_data(parquet_dir: str = "sample_data"):
    d = Path(parquet_dir)
    data = load_data(d)
    topics = load_topics(d)
    return data, topics

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 DBLP EDA")

    data, topics = get_data()
    has_topics = len(topics) > 0

    section = st.radio(
        "Section",
        [
            "📋 Overview",
            "📖 Publications",
            "🏛️ Venues",
            "👤 Authors",
            "🗂️ Topics",
        ],
        disabled=False,
    )

    st.divider()

# ── helpers ───────────────────────────────────────────────────────────────────
def chart(fig, width='stretch'):
    st.plotly_chart(fig, width=width)

def section_header(title: str, description: str = ""):
    st.header(title)
    if description:
        st.caption(description)
    st.divider()

def metric_row(data):
    summary = table_papers_summary(data).iloc[0]
    cols = st.columns(2)
    cols[0].metric("Total papers", f"{int(summary['total_papers']):,}")
    cols[1].metric("Year range", f"{int(summary['year_min'])} – {int(summary['year_max'])}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── 📋 Overview ──────────────────────────────────────────────────────────────
if section == "📋 Overview":
    section_header("Dataset Overview", "High-level statistics and data quality.")

    metric_row(data)
    st.divider()

    st.subheader("Table sizes")
    st.dataframe(table_overview(data), width='stretch', hide_index=True)

    st.subheader("Null values per column")
    null_df = table_null_detail(data)
    col_filter = st.multiselect(
        "Filter by table",
        options=null_df["table"].unique().tolist(),
        default=null_df["table"].unique().tolist(),
    )
    st.dataframe(
        null_df[null_df["table"].isin(col_filter)],
        width='stretch',
        hide_index=True,
    )

    st.subheader("Publication type distribution")
    c1, c2 = st.columns([1, 2])
    with c1:
        chart(type_distribution_pie(data))
    with c2:
        st.dataframe(
            data.papers["type"].value_counts().reset_index().rename(
                columns={"type": "type", "count": "n_papers"}
            ),
            width='stretch',
            hide_index=True,
        )

    st.subheader("Title length analysis")
    chart(title_length_dist(data))


# ── 📖 Publications ────────────────────────────────────────────────────────────
elif section == "📖 Publications":
    section_header("Publications over Time", "Volume, growth and composition by year.")

    metric_row(data)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        chart(publications_per_year(data))
    with c2:
        chart(cumulative_publications(data))

    chart(publications_per_year_by_type(data))
    chart(yoy_growth_rate(data))

    st.subheader("Authors per paper density over time")
    chart(density_heatmap_year_vs_authors(data))


# ── 🏛️ Venues ─────────────────────────────────────────────────────────────────
elif section == "🏛️ Venues":
    section_header("Venues", "Which conferences and journals dominate the field.")

    n_venues = st.slider("Top N venues", 5, 30, 15)

    st.subheader("Top venues by paper count")
    chart(top_venues_bar(data, n=n_venues))

    st.subheader("Conferences vs Journals")
    chart(top_conferences_vs_journals(data, n=n_venues))

    st.subheader("Venue ranking over time")
    chart(venue_bump_chart(data, n=n_venues))

    st.subheader("Top venues — data table")
    st.dataframe(table_top_venues(data, n=n_venues), width='stretch', hide_index=True)

# ── 👤 Authors ─────────────────────────────────────────────────────────────────
elif section == "👤 Authors":
    section_header("Authors", "Productivity, activity spans and career patterns.")

    st.subheader("Authors per paper")
    c1, c2 = st.columns(2)
    with c1:
        chart(authors_per_paper_dist(data))
    with c2:
        chart(authors_per_paper_by_type(data))

    st.subheader("Author productivity")
    n_top = st.slider("Top N prolific authors", 10, 50, 20, key="top_auth")
    chart(top_authors_bar(data, n=n_top))

    chart(author_productivity_dist(data))

    st.subheader("Career span & growth")
    c1, c2 = st.columns(2)
    with c1:
        chart(author_activity_span(data))
    with c2:
        chart(new_authors_per_year(data))

    st.subheader("Top authors — data table")
    st.dataframe(
        table_top_authors(data, n=n_top),
        width='stretch',
        hide_index=True,
    )

# ── 🗂️ Topics ─────────────────────────────────────────────────────────────────
elif section == "🗂️ Topics":
    if not has_topics:
        st.warning("No topics table found in this database. Assign topic_id to papers first.")
        st.stop()

    section_header("Topics", "Research topic distribution, trends and author specialisation.")

    # — overview ——————————————————————————————————————————————————————————————
    st.subheader("Topic overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total topics", len(topics))
    null_pct = data.papers["topic_id"].isnull().mean() * 100
    c2.metric("Papers with topic", f"{100 - null_pct:.1f} %")
    c3.metric("Papers without topic", f"{null_pct:.1f} %")

    c1, c2 = st.columns(2)
    with c1:
        chart(topics_papers_bar(data, topics))
    with c2:
        chart(topics_pie(data, topics))

    chart(topics_treemap(data, topics))

    st.subheader("Topics overview table")
    st.dataframe(table_topics_overview(data, topics), use_container_width=True, hide_index=True)

    st.subheader("Type breakdown per topic")
    st.dataframe(table_topic_type_breakdown(data, topics), use_container_width=True, hide_index=True)

    # — trends ————————————————————————————————————————————————————————————————
    st.subheader("Topic trends over time")
    top_n = st.slider("Top N topics", 3, 15, 8, key="topic_trends_n")

    chart(topics_over_time_line(data, topics, top_n=top_n))
    chart(topics_share_over_time(data, topics, top_n=top_n))
    chart(topic_heatmap_year(data, topics, top_n=top_n))
    chart(topic_bump_chart(data, topics, top_n=top_n))

    c1, c2 = st.columns(2)
    with c1:
        chart(topic_yoy_growth(data, topics, top_n=min(top_n, 8)))
    with c2:
        window = st.slider("Emerging topics window (years)", 2, 5, 3)
        chart(emerging_topics(data, topics, window=window))

    chart(null_topic_over_time(data))
    chart(topic_type_stacked_bar(data, topics))

    # — venues × topics ————————————————————————————————————————————————————————
    st.subheader("Venues × Topics")
    c1, c2 = st.columns(2)
    with c1:
        tv = st.slider("Top venues", 5, 20, 15, key="vt_venues")
        tt = st.slider("Top topics", 3, 15, 10, key="vt_topics")
    chart(venue_topic_heatmap(data, topics, top_venues=tv, top_topics=tt))
    chart(topic_venue_sunburst(data, topics, top_topics=8, top_venues=5))

    # — authors × topics ————————————————————————————————————————————————————————
    st.subheader("Authors × Topics")
    chart(top_authors_per_topic_bar(
        data, topics,
        top_n=st.slider("Top authors per topic", 3, 10, 5),
        top_topics=st.slider("Top topics (authors chart)", 3, 12, 8),
    ))
    chart(author_topic_diversity(
        data, topics,
        min_papers=st.slider("Min papers per author", 2, 20, 5),
    ))
    chart(author_specialisation_scatter(data, topics))

    st.subheader("Topic co-occurrence")
    st.caption("Number of authors who published in both topics.")
    chart(topic_cooccurrence_heatmap(data, topics))

    st.subheader("Top authors per topic — data table")
    n_top_t = st.slider("Top N authors per topic (table)", 3, 20, 10, key="top_auth_topic_tbl")
    st.dataframe(
        table_top_authors_per_topic(data, topics, n=n_top_t),
        width='stretch',
        hide_index=True,
    )