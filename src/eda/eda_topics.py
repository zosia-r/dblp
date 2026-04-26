"""
eda_topics.py — EDA charts and tables that incorporate topic analysis.
Requires topics.parquet and papers.parquet with topic_id column (NULLs handled).

All functions accept a DBData instance (from eda_general.load_data) plus
a topics DataFrame loaded separately.

Usage in Streamlit:
    from pathlib import Path
    from eda_general import load_data
    from eda_topics import load_topics, topics_overview, ...

    data = load_data(Path("sample_data/"))
    topics = load_topics(Path("sample_data/"))
    st.plotly_chart(topics_papers_bar(data, topics))
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eda_general import DBData, COLOR_SEQ, TEMPLATE, table_author_stats

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_topics(parquet_dir: Path) -> pd.DataFrame:
    """Load topics from Parquet. Returns empty DataFrame if file does not exist."""
    path = Path(parquet_dir) / "topics.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["id", "label"])
    return pd.read_parquet(path)


def papers_with_topics(data: DBData, topics: pd.DataFrame) -> pd.DataFrame:
    # ujednolicenie typów przed mergem
    t = topics.copy()
    t["id"] = t["id"].astype("Int64")
    papers = data.papers.copy()
    papers["topic_id"] = pd.to_numeric(papers["topic_id"], errors="coerce").astype("Int64")

    df = papers.merge(
        t.rename(columns={"id": "topic_id", "label": "topic_label"}),
        on="topic_id",
        how="left",
    )
    df["topic_label"] = df["topic_label"].fillna("Unknown")
    return df


# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------

def table_topics_overview(data: DBData, topics: pd.DataFrame) -> pd.DataFrame:
    """Summary table: papers per topic, % of total, year range, top venue."""
    df = papers_with_topics(data, topics)
    total = len(df)
    agg = (
        df.groupby("topic_label")
        .agg(
            n_papers=("id", "count"),
            year_min=("year", "min"),
            year_max=("year", "max"),
        )
        .reset_index()
    )
    agg["pct_total"] = (agg["n_papers"] / total * 100).round(2)

    # top venue per topic
    top_venue = (
        df.dropna(subset=["venue"])
        .groupby(["topic_label", "venue"]).size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("topic_label")[["topic_label", "venue"]]
        .rename(columns={"venue": "top_venue"})
    )
    return (
        agg.merge(top_venue, on="topic_label", how="left")
        .sort_values("n_papers", ascending=False)
        .reset_index(drop=True)
    )


def table_topic_type_breakdown(data: DBData, topics: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation: topics × publication types (paper counts)."""
    df = papers_with_topics(data, topics)
    return (
        df.groupby(["topic_label", "type"]).size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"topic_label": "topic"})
    )


def table_top_authors_per_topic(
    data: DBData, topics: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
    """Top N authors per topic by paper count."""
    df = papers_with_topics(data, topics)
    merged = (
        data.paper_authors
        .merge(df[["id", "topic_label"]], left_on="paper_id", right_on="id")
        .merge(data.authors[["id", "primary_name"]], left_on="author_id", right_on="id")
        .groupby(["topic_label", "primary_name"])["paper_id"].count()
        .reset_index(name="n_papers")
    )
    return (
        merged.sort_values(["topic_label", "n_papers"], ascending=[True, False])
        .groupby("topic_label").head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------

def topics_papers_bar(data: DBData, topics: pd.DataFrame) -> go.Figure:
    """Horizontal bar: total papers per topic, sorted descending."""
    df = papers_with_topics(data, topics)
    counts = (
        df.groupby("topic_label").size().rename("n_papers")
        .reset_index().sort_values("n_papers")
    )
    fig = px.bar(
        counts, x="n_papers", y="topic_label", orientation="h",
        title="Papers per Topic",
        labels={"n_papers": "Number of papers", "topic_label": "Topic"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(height=max(400, len(counts) * 22))
    return fig


def topics_pie(data: DBData, topics: pd.DataFrame) -> go.Figure:
    """Pie chart: share of each topic in total publications."""
    df = papers_with_topics(data, topics)
    counts = df["topic_label"].value_counts().reset_index()
    counts.columns = ["topic", "n_papers"]
    fig = px.pie(
        counts, names="topic", values="n_papers",
        title="Topic Share of Total Publications",
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def topics_treemap(data: DBData, topics: pd.DataFrame) -> go.Figure:
    """Treemap: topic → publication type → paper count."""
    df = papers_with_topics(data, topics)
    agg = (
        df.groupby(["topic_label", "type"]).size()
        .reset_index(name="n_papers")
    )
    agg["root"] = "All"
    fig = px.treemap(
        agg, path=["root", "topic_label", "type"], values="n_papers",
        title="Treemap: Topic → Publication Type",
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


# ---------------------------------------------------------------------------
# TRENDS OVER TIME
# ---------------------------------------------------------------------------

def topics_over_time_line(data: DBData, topics: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Line chart: paper count per year for top N topics."""
    df = papers_with_topics(data, topics)
    top = (
        df[df["topic_label"] != "Unknown"]
        .groupby("topic_label").size().nlargest(top_n).index
    )
    trend = (
        df[df["topic_label"].isin(top)]
        .groupby(["year", "topic_label"]).size().rename("n_papers").reset_index()
    )
    fig = px.line(
        trend, x="year", y="n_papers", color="topic_label",
        title=f"Top {top_n} Topics — Papers per Year",
        labels={"n_papers": "Papers", "year": "Year", "topic_label": "Topic"},
        markers=True, template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def topics_share_over_time(data: DBData, topics: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Stacked area (normalised to 100%): topic share over time."""
    df = papers_with_topics(data, topics)
    top = (
        df[df["topic_label"] != "Unknown"]
        .groupby("topic_label").size().nlargest(top_n).index
    )
    trend = (
        df[df["topic_label"].isin(top)]
        .groupby(["year", "topic_label"]).size().rename("n_papers").reset_index()
    )
    # normalise per year
    trend["total_year"] = trend.groupby("year")["n_papers"].transform("sum")
    trend["share_pct"] = trend["n_papers"] / trend["total_year"] * 100

    fig = px.area(
        trend, x="year", y="share_pct", color="topic_label",
        title=f"Topic Share over Time (top {top_n}, % of year total)",
        labels={"share_pct": "Share (%)", "year": "Year", "topic_label": "Topic"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def topic_heatmap_year(data: DBData, topics: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Heatmap: top N topics × year, coloured by paper count."""
    df = papers_with_topics(data, topics)
    top = (
        df[df["topic_label"] != "Unknown"]
        .groupby("topic_label").size().nlargest(top_n).index
    )
    hm = (
        df[df["topic_label"].isin(top)]
        .groupby(["topic_label", "year"]).size()
        .unstack(fill_value=0)
    )
    fig = go.Figure(go.Heatmap(
        z=hm.values,
        x=hm.columns.tolist(),
        y=hm.index.tolist(),
        colorscale="Blues",
        colorbar=dict(title="papers"),
    ))
    fig.update_layout(
        title=f"Top {top_n} Topics × Year Heatmap",
        xaxis_title="Year", yaxis_title="Topic",
        template=TEMPLATE, height=500,
    )
    return fig


def topic_bump_chart(data: DBData, topics: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Bump chart: ranking of top N topics over time."""
    df = papers_with_topics(data, topics)
    top = (
        df[df["topic_label"] != "Unknown"]
        .groupby("topic_label").size().nlargest(top_n).index
    )
    ranked = (
        df[df["topic_label"].isin(top)]
        .groupby(["year", "topic_label"]).size().rename("n_papers").reset_index()
    )
    ranked["rank"] = (
        ranked.groupby("year")["n_papers"]
        .rank(ascending=False, method="min").astype(int)
    )
    fig = go.Figure()
    for i, label in enumerate(top):
        d = ranked[ranked["topic_label"] == label].sort_values("year")
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["rank"], mode="lines+markers",
            name=label,
            line=dict(width=2, color=COLOR_SEQ[i % len(COLOR_SEQ)]),
            marker=dict(size=8),
        ))
    fig.update_yaxes(autorange="reversed", title="Rank (1 = most papers)")
    fig.update_layout(
        title=f"Top {top_n} Topics — Ranking over Time",
        xaxis_title="Year", template=TEMPLATE,
    )
    return fig


def topic_yoy_growth(data: DBData, topics: pd.DataFrame, top_n: int = 6) -> go.Figure:
    """Heatmap: year-on-year growth rate per topic for top N topics."""
    df = papers_with_topics(data, topics)
    top = (
        df[df["topic_label"] != "Unknown"]
        .groupby("topic_label").size().nlargest(top_n).index
    )
    trend = (
        df[df["topic_label"].isin(top)]
        .groupby(["topic_label", "year"]).size().rename("n").reset_index()
        .sort_values(["topic_label", "year"])
    )
    trend["growth_pct"] = (
        trend.groupby("topic_label")["n"].pct_change() * 100
    )
    pivot = trend.pivot(index="topic_label", columns="year", values="growth_pct").dropna(axis=1, how="all")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="YoY growth (%)"),
    ))
    fig.update_layout(
        title=f"Year-on-Year Growth Rate by Topic (top {top_n})",
        xaxis_title="Year", yaxis_title="Topic",
        template=TEMPLATE, height=400,
    )
    return fig


def emerging_topics(data: DBData, topics: pd.DataFrame, window: int = 3) -> go.Figure:
    """Bar: topics with highest growth in the last N years vs previous N years."""
    df = papers_with_topics(data, topics)
    df = df[df["topic_label"] != "Unknown"]
    max_year = df["year"].max()
    recent = df[df["year"] > max_year - window].groupby("topic_label").size().rename("recent")
    previous = df[(df["year"] > max_year - 2 * window) & (df["year"] <= max_year - window)].groupby("topic_label").size().rename("previous")
    comb = pd.concat([recent, previous], axis=1).fillna(0)
    comb["growth"] = ((comb["recent"] - comb["previous"]) / (comb["previous"] + 1) * 100).round(1)
    comb = comb.sort_values("growth", ascending=False).head(15).reset_index()
    fig = px.bar(
        comb.sort_values("growth"), x="growth", y="topic_label", orientation="h",
        title=f"Emerging Topics: Growth in Last {window} Years vs Previous {window} Years (%)",
        labels={"growth": "Growth (%)", "topic_label": "Topic"},
        color="growth", color_continuous_scale="RdYlGn",
        template=TEMPLATE,
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ---------------------------------------------------------------------------
# VENUES × TOPICS
# ---------------------------------------------------------------------------

def venue_topic_heatmap(data: DBData, topics: pd.DataFrame,
                         top_venues: int = 15, top_topics: int = 10) -> go.Figure:
    """Heatmap: top venues × top topics, coloured by paper count."""
    df = papers_with_topics(data, topics)
    tv = (
        df.dropna(subset=["venue"])
        .groupby("venue").size().nlargest(top_venues).index
    )
    tt = (
        df[df["topic_label"] != "Unknown"]
        .groupby("topic_label").size().nlargest(top_topics).index
    )
    hm = (
        df[df["venue"].isin(tv) & df["topic_label"].isin(tt)]
        .groupby(["venue", "topic_label"]).size()
        .unstack(fill_value=0)
    )
    fig = go.Figure(go.Heatmap(
        z=hm.values,
        x=hm.columns.tolist(),
        y=hm.index.tolist(),
        colorscale="YlOrRd",
        colorbar=dict(title="papers"),
    ))
    fig.update_layout(
        title=f"Top Venues × Top Topics Heatmap",
        xaxis_title="Topic", yaxis_title="Venue",
        template=TEMPLATE, height=520,
    )
    return fig


def topic_venue_sunburst(data: DBData, topics: pd.DataFrame,
                          top_topics: int = 8, top_venues: int = 5) -> go.Figure:
    """Sunburst: topic → top venues."""
    df = papers_with_topics(data, topics)
    tt = (
        df[df["topic_label"] != "Unknown"]
        .groupby("topic_label").size().nlargest(top_topics).index
    )
    df_f = df[df["topic_label"].isin(tt) & df["venue"].notna()]
    # keep top venues per topic
    top_v_per_t = (
        df_f.groupby(["topic_label", "venue"]).size().reset_index(name="n_papers")
        .sort_values(["topic_label", "n_papers"], ascending=[True, False])
        .groupby("topic_label").head(top_venues)
    )
    top_v_per_t["root"] = "All"
    fig = px.sunburst(
        top_v_per_t, path=["root", "topic_label", "venue"], values="n_papers",
        title=f"Sunburst: Top {top_topics} Topics → Top {top_venues} Venues each",
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


# ---------------------------------------------------------------------------
# AUTHORS × TOPICS
# ---------------------------------------------------------------------------

def top_authors_per_topic_bar(data: DBData, topics: pd.DataFrame,
                               top_n: int = 5, top_topics: int = 8) -> go.Figure:
    """Grouped bar: top N authors per topic (for top M topics)."""
    df = papers_with_topics(data, topics)
    top_t = (
        df[df["topic_label"] != "Unknown"]
        .groupby("topic_label").size().nlargest(top_topics).index
    )
    merged = (
        data.paper_authors
        .merge(df[["id", "topic_label"]], left_on="paper_id", right_on="id")
        .merge(data.authors[["id", "primary_name"]], left_on="author_id", right_on="id")
        .groupby(["topic_label", "primary_name"])["paper_id"].count()
        .reset_index(name="n_papers")
    )
    top_auth = (
        merged[merged["topic_label"].isin(top_t)]
        .sort_values(["topic_label", "n_papers"], ascending=[True, False])
        .groupby("topic_label").head(top_n)
    )
    fig = px.bar(
        top_auth, x="n_papers", y="primary_name", color="topic_label",
        orientation="h", barmode="group",
        title=f"Top {top_n} Authors per Topic (top {top_topics} topics)",
        labels={"n_papers": "Papers", "primary_name": "Author", "topic_label": "Topic"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
        height=max(500, top_n * top_topics * 18),
    )
    return fig


def author_topic_diversity(data: DBData, topics: pd.DataFrame, min_papers: int = 5) -> go.Figure:
    """Histogram: distribution of number of distinct topics per author."""
    df = papers_with_topics(data, topics)
    df = df[df["topic_label"] != "Unknown"]
    diversity = (
        data.paper_authors
        .merge(df[["id", "topic_label"]], left_on="paper_id", right_on="id")
        .groupby("author_id")["topic_label"].nunique().rename("n_topics")
        .reset_index()
    )
    # filter to authors with at least min_papers
    prod = data.paper_authors.groupby("author_id")["paper_id"].count().rename("n_papers")
    diversity = diversity.merge(prod, on="author_id")
    diversity = diversity[diversity["n_papers"] >= min_papers]
    fig = px.histogram(
        diversity, x="n_topics",
        title=f"Author Topic Diversity (authors with ≥{min_papers} papers)",
        labels={"n_topics": "Distinct topics", "count": "Authors"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def author_specialisation_scatter(data: DBData, topics: pd.DataFrame,
                                   min_papers: int = 10) -> go.Figure:
    """Scatter: total papers vs topic diversity, sized by active years."""
    df = papers_with_topics(data, topics)
    df = df[df["topic_label"] != "Unknown"]
    diversity = (
        data.paper_authors
        .merge(df[["id", "topic_label"]], left_on="paper_id", right_on="id")
        .groupby("author_id")["topic_label"].nunique().rename("n_topics")
        .reset_index()
    )
    stats = table_author_stats(data)
    merged = stats.merge(diversity, on="author_id")
    merged = merged[merged["n_papers"] >= min_papers].sample(min(3000, len(merged)), random_state=42)
    fig = px.scatter(
        merged, x="n_topics", y="n_papers",
        size="active_years", size_max=15,
        opacity=0.5,
        title=f"Author Specialisation: Papers vs Topic Diversity (≥{min_papers} papers)",
        labels={"n_topics": "Distinct topics", "n_papers": "Total papers",
                "active_years": "Active years"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def topic_cooccurrence_heatmap(data: DBData, topics: pd.DataFrame) -> go.Figure:
    """
    Heatmap: how often authors publish across pairs of topics.
    Cell (i,j) = number of authors who published in both topic i and topic j.
    """
    df = papers_with_topics(data, topics)
    df = df[df["topic_label"] != "Unknown"]
    author_topics = (
        data.paper_authors
        .merge(df[["id", "topic_label"]], left_on="paper_id", right_on="id")
        [["author_id", "topic_label"]].drop_duplicates()
    )
    topic_labels = sorted(author_topics["topic_label"].unique())
    n = len(topic_labels)
    idx = {t: i for i, t in enumerate(topic_labels)}
    mat = [[0] * n for _ in range(n)]
    for _, grp in author_topics.groupby("author_id")["topic_label"]:
        labels = list(grp)
        for a in labels:
            for b in labels:
                mat[idx[a]][idx[b]] += 1
    fig = go.Figure(go.Heatmap(
        z=mat, x=topic_labels, y=topic_labels,
        colorscale="Blues",
        colorbar=dict(title="authors"),
    ))
    fig.update_layout(
        title="Topic Co-occurrence: Authors Publishing in Both Topics",
        template=TEMPLATE, height=550,
    )
    return fig


# ---------------------------------------------------------------------------
# PUBLICATION TYPE × TOPIC
# ---------------------------------------------------------------------------

def topic_type_stacked_bar(data: DBData, topics: pd.DataFrame) -> go.Figure:
    """Stacked bar: for each topic, breakdown by publication type."""
    df = papers_with_topics(data, topics)
    agg = (
        df[df["topic_label"] != "Unknown"]
        .groupby(["topic_label", "type"]).size().reset_index(name="n_papers")
    )
    total = agg.groupby("topic_label")["n_papers"].transform("sum")
    agg["pct"] = agg["n_papers"] / total * 100
    fig = px.bar(
        agg.sort_values("n_papers", ascending=False),
        x="topic_label", y="pct", color="type",
        barmode="stack",
        title="Publication Type Breakdown per Topic (%)",
        labels={"pct": "Share (%)", "topic_label": "Topic", "type": "Type"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(xaxis_tickangle=-30)
    return fig


def null_topic_over_time(data: DBData) -> go.Figure:
    """Line chart: % of papers with missing topic_id per year."""
    p = data.papers.copy()
    p["no_topic"] = p["topic_id"].isnull().astype(int)
    agg = (
        p.groupby("year")
        .agg(n_papers=("id", "count"), n_null=("no_topic", "sum"))
        .reset_index()
    )
    agg["pct_null"] = agg["n_null"] / agg["n_papers"] * 100
    fig = px.line(
        agg, x="year", y="pct_null",
        title="Papers with Missing Topic (%) over Time",
        labels={"pct_null": "% missing topic", "year": "Year"},
        markers=True, template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig