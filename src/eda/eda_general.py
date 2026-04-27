"""
eda_general.py — EDA charts and tables without topics.
All functions accept a DBData container populated from Parquet files.
Returns plotly Figure objects or pandas DataFrames ready for Streamlit.

Usage in Streamlit:
    from pathlib import Path
    from eda_general import load_data, publications_per_year, ...

    data = load_data(Path("sample_data/"))
    st.plotly_chart(publications_per_year(data))
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# colour palette & template
# ---------------------------------------------------------------------------

TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Safe


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

class DBData:
    """Container for all tables loaded from the database."""
    papers: pd.DataFrame
    authors: pd.DataFrame
    aliases: pd.DataFrame
    paper_authors: pd.DataFrame


def _decode_binary_int(value):
    """Decode little-endian 8-byte binary integers to Python int."""
    if isinstance(value, memoryview):
        value = value.tobytes()

    if isinstance(value, (bytes, bytearray)) and len(value) == 8:
        return int.from_bytes(value, byteorder="little", signed=False)

    if isinstance(value, np.void):
        raw = bytes(value)
        if len(raw) == 8:
            return int.from_bytes(raw, byteorder="little", signed=False)

    return value


def _normalize_binary_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert binary-encoded integers in object columns to Python ints."""
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].map(_decode_binary_int)
    return df


def load_data(parquet_dir: Path) -> DBData:
    """Load all tables from Parquet files into a DBData container.

    Expected files (as produced by sample_db.py):
        papers.parquet, authors.parquet, author_aliases.parquet, paper_authors.parquet
    """
    d = Path(parquet_dir)
    data = DBData()
    data.papers = _normalize_binary_int_columns(pd.read_parquet(d / "papers.parquet"))
    data.authors = _normalize_binary_int_columns(pd.read_parquet(d / "authors.parquet"))
    data.aliases = _normalize_binary_int_columns(pd.read_parquet(d / "author_aliases.parquet"))
    data.paper_authors = _normalize_binary_int_columns(pd.read_parquet(d / "paper_authors.parquet"))

    return data


# ---------------------------------------------------------------------------
# TABLE helpers
# ---------------------------------------------------------------------------

def table_overview(data: DBData) -> pd.DataFrame:
    """Row counts, column counts, nulls and uniques per table."""
    rows = []
    for name, df in [
        ("papers", data.papers),
        ("authors", data.authors),
        ("author_aliases", data.aliases),
        ("paper_authors", data.paper_authors),
    ]:
        rows.append({
            "table": name,
            "rows": len(df),
            "columns": df.shape[1],
            "total_nulls": int(df.isnull().sum().sum()),
            "null_pct": round(df.isnull().mean().mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def table_null_detail(data: DBData) -> pd.DataFrame:
    """Per-column null statistics across all tables."""
    rows = []
    for name, df in [
        ("papers", data.papers),
        ("authors", data.authors),
        ("author_aliases", data.aliases),
        ("paper_authors", data.paper_authors),
    ]:
        for col in df.columns:
            n_null = int(df[col].isnull().sum())
            rows.append({
                "table": name,
                "column": col,
                "dtype": str(df[col].dtype),
                "n_null": n_null,
                "pct_null": round(n_null / len(df) * 100, 2),
                "n_unique": int(df[col].nunique()),
            })
    return pd.DataFrame(rows)


def table_papers_summary(data: DBData) -> pd.DataFrame:
    """High-level summary statistics for the papers table."""
    p = data.papers
    pa = data.paper_authors
    author_counts = pa.groupby("paper_id")["author_id"].count()
    return pd.DataFrame([{
        "total_papers": len(p),
        "year_min": int(p["year"].min()),
        "year_max": int(p["year"].max()),
        "unique_venues": int(p["venue"].nunique()),
        "missing_venue_pct": round(p["venue"].isnull().mean() * 100, 2),
        "unique_types": int(p["type"].nunique()),
        "solo_papers_pct": round((author_counts == 1).mean() * 100, 2),
        "median_authors_per_paper": float(author_counts.median()),
    }])


def table_top_venues(data: DBData, n: int = 20) -> pd.DataFrame:
    """Top N venues by paper count with type breakdown."""
    return (
        data.papers.dropna(subset=["venue"])
        .groupby(["venue", "type"])
        .size()
        .reset_index(name="n_papers")
        .assign(total=lambda d: d.groupby("venue")["n_papers"].transform("sum"))
        .sort_values(["total", "venue"], ascending=[False, True])
        .groupby("venue")
        .head(99)  # keep all types per venue
        .pipe(lambda d: d[d["venue"].isin(
            d.groupby("venue")["n_papers"].sum().nlargest(n).index
        )])
        .drop(columns="total")
        .reset_index(drop=True)
    )


def table_top_authors(data: DBData, n: int = 30) -> pd.DataFrame:
    """Top N authors by paper count."""
    return (
        data.paper_authors
        .groupby("author_id")["paper_id"].count().rename("n_papers")
        .reset_index()
        .merge(data.authors[["id", "primary_name"]], left_on="author_id", right_on="id")
        .drop(columns="id")
        .sort_values("n_papers", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def table_author_stats(data: DBData) -> pd.DataFrame:
    """Per-author aggregate statistics."""
    author_years = (
        data.paper_authors
        .merge(data.papers[["id", "year", "venue"]], left_on="paper_id", right_on="id")
        .groupby("author_id")
        .agg(
            n_papers=("paper_id", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_venues=("venue", "nunique"),
        )
        .reset_index()
    )
    author_years["active_years"] = author_years["last_year"] - author_years["first_year"] + 1
    return author_years.merge(
        data.authors[["id", "primary_name"]], left_on="author_id", right_on="id"
    ).drop(columns="id")


# ---------------------------------------------------------------------------
# PUBLICATIONS OVER TIME
# ---------------------------------------------------------------------------

def publications_per_year(data: DBData) -> go.Figure:
    """Bar chart: total papers published per year."""
    by_year = data.papers.groupby("year").size().rename("n_papers").reset_index()
    fig = px.bar(
        by_year, x="year", y="n_papers",
        title="Publications per Year",
        labels={"n_papers": "Number of papers", "year": "Year"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(bargap=0.1)
    return fig


def publications_per_year_by_type(data: DBData) -> go.Figure:
    """Stacked area chart: papers per year broken down by publication type."""
    by_year_type = (
        data.papers.groupby(["year", "type"]).size().rename("n_papers").reset_index()
    )
    fig = px.area(
        by_year_type, x="year", y="n_papers", color="type",
        title="Publications per Year by Type",
        labels={"n_papers": "Number of papers", "year": "Year", "type": "Type"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def type_distribution_pie(data: DBData) -> go.Figure:
    """Pie chart: share of each publication type."""
    counts = data.papers["type"].value_counts().reset_index()
    counts.columns = ["type", "n_papers"]
    fig = px.pie(
        counts, names="type", values="n_papers",
        title="Publication Type Distribution",
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def cumulative_publications(data: DBData) -> go.Figure:
    """Line chart: cumulative paper count over time."""
    by_year = (
        data.papers.groupby("year").size().rename("n_papers")
        .sort_index().cumsum().reset_index()
    )
    fig = px.line(
        by_year, x="year", y="n_papers",
        title="Cumulative Publications over Time",
        labels={"n_papers": "Total papers (cumulative)", "year": "Year"},
        markers=True,
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def yoy_growth_rate(data: DBData) -> go.Figure:
    """Bar chart: year-on-year growth rate of publications."""
    by_year = data.papers.groupby("year").size().rename("n").sort_index()
    growth = (by_year.pct_change() * 100).rename("growth_pct").reset_index()
    growth = growth.dropna()
    fig = px.bar(
        growth, x="year", y="growth_pct",
        title="Year-on-Year Publication Growth Rate (%)",
        labels={"growth_pct": "Growth (%)", "year": "Year"},
        template=TEMPLATE,
        color="growth_pct",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ---------------------------------------------------------------------------
# VENUES
# ---------------------------------------------------------------------------

def top_venues_bar(data: DBData, n: int = 15) -> go.Figure:
    """Horizontal bar chart: top N venues by total paper count, coloured by type."""
    df = table_top_venues(data, n)
    fig = px.bar(
        df.sort_values("n_papers"),
        x="n_papers", y="venue", color="type", orientation="h",
        barmode="stack",
        title=f"Top {n} Venues by Paper Count",
        labels={"n_papers": "Number of papers", "venue": "Venue"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(height=500)
    return fig


def top_conferences_vs_journals(data: DBData, n: int = 15) -> go.Figure:
    """Side-by-side bars: top conferences and top journals."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Top {n} Conferences", f"Top {n} Journals"),
    )
    for col_idx, ptype in enumerate(["inproceedings", "article"], start=1):
        sub = (
            data.papers[(data.papers["type"] == ptype) & data.papers["venue"].notna()]
            .groupby("venue").size().nlargest(n).reset_index(name="count")
            .sort_values("count")
        )
        fig.add_trace(
            go.Bar(x=sub["count"], y=sub["venue"], orientation="h",
                   marker_color=COLOR_SEQ[col_idx - 1], name=ptype),
            row=1, col=col_idx,
        )
    fig.update_layout(template=TEMPLATE, height=520,
                      title_text="Top Venues per Publication Type", showlegend=False)
    return fig


def venue_heatmap(data: DBData, n: int = 20) -> go.Figure:
    """Heatmap: top N venues × year, coloured by paper count."""
    top_n = (
        data.papers.dropna(subset=["venue"])
        .groupby("venue").size().nlargest(n).index
    )
    hm = (
        data.papers[data.papers["venue"].isin(top_n)]
        .groupby(["venue", "year"]).size()
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
        title=f"Top {n} Venues × Year Heatmap",
        xaxis_title="Year", yaxis_title="Venue",
        template=TEMPLATE, height=550,
    )
    return fig


def venue_bump_chart(data: DBData, n: int = 10) -> go.Figure:
    """Bump chart: ranking of top N venues over time."""
    top_n = (
        data.papers.dropna(subset=["venue"])
        .groupby("venue").size().nlargest(n).index
    )
    ranked = (
        data.papers[data.papers["venue"].isin(top_n)]
        .groupby(["year", "venue"]).size().rename("n_papers").reset_index()
    )
    ranked["rank"] = (
        ranked.groupby("year")["n_papers"]
        .rank(ascending=False, method="min").astype(int)
    )
    fig = go.Figure()
    for i, venue in enumerate(top_n):
        d = ranked[ranked["venue"] == venue].sort_values("year")
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["rank"], mode="lines+markers",
            name=venue, line=dict(width=2, color=COLOR_SEQ[i % len(COLOR_SEQ)]),
            marker=dict(size=8),
        ))
    fig.update_yaxes(autorange="reversed", title="Rank (1 = most papers)")
    fig.update_layout(
        title=f"Top {n} Venues — Ranking over Time",
        xaxis_title="Year", template=TEMPLATE,
    )
    return fig


def venue_treemap(data: DBData, n: int = 30) -> go.Figure:
    """Treemap: top N venues → publication type → paper count."""
    top_n = (
        data.papers.dropna(subset=["venue"])
        .groupby("venue").size().nlargest(n).index
    )
    df = (
        data.papers[data.papers["venue"].isin(top_n)]
        .groupby(["venue", "type"]).size().reset_index(name="n_papers")
    )
    df["root"] = "All"
    fig = px.treemap(
        df, path=["root", "venue", "type"], values="n_papers",
        title=f"Treemap: Top {n} Venues → Type",
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def venue_sankey(data: DBData, n: int = 6) -> go.Figure:
    """Sankey: author overlap between top N venues."""
    top_n = (
        data.papers.dropna(subset=["venue"])
        .groupby("venue").size().nlargest(n).index.tolist()
    )
    va = (
        data.paper_authors
        .merge(data.papers[["id", "venue"]], left_on="paper_id", right_on="id")
        [["author_id", "venue"]].drop_duplicates()
    )
    va = va[va["venue"].isin(top_n)]
    multi = va.groupby("author_id")["venue"].nunique()
    multi = multi[multi >= 2].index
    flows = va[va["author_id"].isin(multi)]
    pairs = (
        flows.merge(flows, on="author_id")
        .query("venue_x < venue_y")
        .groupby(["venue_x", "venue_y"]).size().reset_index(name="n_authors")
    )
    idx = {v: i for i, v in enumerate(top_n)}
    fig = go.Figure(go.Sankey(
        node=dict(label=top_n, pad=20, thickness=20,
                  color=COLOR_SEQ[:len(top_n)]),
        link=dict(
            source=pairs["venue_x"].map(idx),
            target=pairs["venue_y"].map(idx),
            value=pairs["n_authors"],
        ),
    ))
    fig.update_layout(title=f"Author Overlap between Top {n} Venues", template=TEMPLATE)
    return fig


def unique_venues_per_year(data: DBData) -> go.Figure:
    """Bar chart: number of unique venues active each year."""
    uv = (
        data.papers.dropna(subset=["venue"])
        .groupby("year")["venue"].nunique().reset_index(name="unique_venues")
    )
    fig = px.bar(
        uv, x="year", y="unique_venues",
        title="Unique Venues per Year",
        labels={"unique_venues": "Unique venues", "year": "Year"},
        color="unique_venues", color_continuous_scale="Blues",
        template=TEMPLATE,
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ---------------------------------------------------------------------------
# AUTHORS
# ---------------------------------------------------------------------------

def authors_per_paper_dist(data: DBData) -> go.Figure:
    """Histogram: distribution of authors per paper (capped at 15)."""
    counts = (
        data.paper_authors.groupby("paper_id")["author_id"]
        .count().clip(upper=15).rename("n_authors")
    )
    fig = px.histogram(
        counts.reset_index(), x="n_authors", nbins=15,
        title="Distribution of Authors per Paper (capped at 15)",
        labels={"n_authors": "Authors per paper", "count": "Papers"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def authors_per_paper_by_type(data: DBData) -> go.Figure:
    """Violin: authors per paper split by publication type."""
    df = (
        data.paper_authors.groupby("paper_id")["author_id"]
        .count().rename("n_authors").reset_index()
        .merge(data.papers[["id", "type"]], left_on="paper_id", right_on="id")
    )
    df["n_authors_capped"] = df["n_authors"].clip(upper=20)
    fig = px.violin(
        df, x="type", y="n_authors_capped", box=True,
        title="Authors per Paper by Publication Type (capped at 20)",
        labels={"n_authors_capped": "Authors per paper", "type": "Type"},
        template=TEMPLATE,
        color="type", color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def median_authors_per_year(data: DBData) -> go.Figure:
    """Line chart: median authors per paper over time, by type."""
    df = (
        data.paper_authors.groupby("paper_id")["author_id"]
        .count().rename("n_authors").reset_index()
        .merge(data.papers[["id", "year", "type"]], left_on="paper_id", right_on="id")
    )
    med = df.groupby(["year", "type"])["n_authors"].median().reset_index()
    fig = px.line(
        med, x="year", y="n_authors", color="type",
        title="Median Authors per Paper over Time",
        labels={"n_authors": "Median authors", "year": "Year"},
        markers=True, template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def top_authors_bar(data: DBData, n: int = 20) -> go.Figure:
    """Horizontal bar: top N most prolific authors."""
    df = table_top_authors(data, n).sort_values("n_papers")
    fig = px.bar(
        df, x="n_papers", y="primary_name", orientation="h",
        title=f"Top {n} Most Prolific Authors",
        labels={"n_papers": "Number of papers", "primary_name": "Author"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(height=max(400, n * 25))
    return fig


def author_productivity_dist(data: DBData) -> go.Figure:
    """Log-scale bar: distribution of authors by paper count (capped at 50)."""
    prod = (
        data.paper_authors.groupby("author_id")["paper_id"]
        .count().clip(upper=50).rename("n_papers")
        .value_counts().sort_index().reset_index()
    )
    prod.columns = ["n_papers", "n_authors"]
    fig = px.bar(
        prod, x="n_papers", y="n_authors",
        title="Author Productivity Distribution (capped at 50)",
        labels={"n_papers": "Papers published", "n_authors": "Number of authors"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
        log_y=True,
    )
    return fig


def author_activity_span(data: DBData) -> go.Figure:
    """Histogram: how many years each author was active."""
    ay = (
        data.paper_authors
        .merge(data.papers[["id", "year"]], left_on="paper_id", right_on="id")
        .groupby("author_id")["year"]
        .agg(first=("min"), last=("max"))
        .reset_index()
    )
    ay["active_years"] = ay["last"] - ay["first"] + 1
    fig = px.histogram(
        ay, x="active_years",
        title="Author Activity Span (years)",
        labels={"active_years": "Active years", "count": "Authors"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def new_authors_per_year(data: DBData) -> go.Figure:
    """Bar chart: number of authors publishing for the first time each year."""
    debut = (
        data.paper_authors
        .merge(data.papers[["id", "year"]], left_on="paper_id", right_on="id")
        .groupby("author_id")["year"].min().rename("debut_year").reset_index()
    )
    counts = debut["debut_year"].value_counts().sort_index().reset_index()
    counts.columns = ["year", "new_authors"]
    fig = px.bar(
        counts, x="year", y="new_authors",
        title="New Authors per Year (first publication)",
        labels={"new_authors": "New authors", "year": "Year"},
        color="new_authors", color_continuous_scale="Teal",
        template=TEMPLATE,
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def author_retention_funnel(data: DBData) -> go.Figure:
    """Funnel: how many authors published in at least N distinct years."""
    year_counts = (
        data.paper_authors
        .merge(data.papers[["id", "year"]], left_on="paper_id", right_on="id")
        [["author_id", "year"]].drop_duplicates()
        .groupby("author_id")["year"].nunique()
    )
    funnel = pd.DataFrame({
        "min_active_years": range(1, 12),
        "n_authors": [(year_counts >= n).sum() for n in range(1, 12)],
    })
    fig = px.funnel(
        funnel, x="n_authors", y="min_active_years",
        title="Author Retention: Active in at Least N Years",
        labels={"n_authors": "Authors", "min_active_years": "Min active years"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def author_scatter_papers_vs_years(data: DBData) -> go.Figure:
    """Scatter with marginals: papers published vs active years (sample)."""
    stats = table_author_stats(data)
    sample = (
        stats[stats["n_papers"] >= 2]
        .sample(min(4000, len(stats)), random_state=42)
    )
    sample["n_papers_capped"] = sample["n_papers"].clip(upper=100)
    fig = px.scatter(
        sample, x="active_years", y="n_papers_capped",
        marginal_x="histogram", marginal_y="histogram",
        opacity=0.4,
        title="Papers Published vs Active Years (≥2 papers, capped at 100)",
        labels={"active_years": "Active years", "n_papers_capped": "Papers (capped)"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def author_parallel_coordinates(data: DBData) -> go.Figure:
    """Parallel coordinates: author profile (papers, active years, venues)."""
    stats = table_author_stats(data)
    sample = (
        stats[stats["n_papers"] >= 3]
        .sample(min(5000, len(stats)), random_state=42)
    )
    fig = px.parallel_coordinates(
        sample,
        dimensions=["n_papers", "active_years", "n_venues"],
        color="n_papers",
        color_continuous_scale="Viridis",
        title="Author Profiles: Papers × Active Years × Venues",
        template=TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# CO-AUTHORSHIP
# ---------------------------------------------------------------------------

def _build_edges(data: DBData) -> pd.DataFrame:
    """Build co-authorship edge list with weights."""
    multi = (
        data.paper_authors.groupby("paper_id")["author_id"]
        .apply(list).reset_index()
    )
    multi = multi[multi["author_id"].map(len) >= 2]
    records = [
        (a, b)
        for authors_list in multi["author_id"]
        for a, b in combinations(sorted(authors_list), 2)
    ]
    edges = (
        pd.DataFrame(records, columns=["author_a", "author_b"])
        .groupby(["author_a", "author_b"]).size().rename("weight")
        .reset_index()
    )
    return edges


def coauthorship_weight_dist(data: DBData) -> go.Figure:
    """Bar: distribution of co-authorship pair weights (capped at 20)."""
    edges = _build_edges(data)
    dist = (
        edges["weight"].clip(upper=20)
        .value_counts().sort_index().reset_index()
    )
    dist.columns = ["weight", "n_pairs"]
    fig = px.bar(
        dist, x="weight", y="n_pairs",
        title="Co-authorship Weight Distribution (capped at 20)",
        labels={"weight": "Shared papers", "n_pairs": "Author pairs"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


def top_authors_by_degree(data: DBData, n: int = 20) -> go.Figure:
    """Horizontal bar: top N authors by co-authorship degree (unique collaborators)."""
    edges = _build_edges(data)
    from collections import Counter
    degree: Counter = Counter()
    for _, row in edges.iterrows():
        degree[row["author_a"]] += 1
        degree[row["author_b"]] += 1
    top = (
        pd.DataFrame(degree.most_common(n), columns=["author_id", "degree"])
        .merge(data.authors[["id", "primary_name"]], left_on="author_id", right_on="id")
        .sort_values("degree")
    )
    fig = px.bar(
        top, x="degree", y="primary_name", orientation="h",
        title=f"Top {n} Authors by Collaboration Degree",
        labels={"degree": "Unique collaborators", "primary_name": "Author"},
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(height=max(400, n * 25))
    return fig


def avg_collaborators_per_year(data: DBData) -> go.Figure:
    """Line chart: average number of collaborators per author per year."""
    df = (
        data.paper_authors
        .merge(data.papers[["id", "year"]], left_on="paper_id", right_on="id")
    )
    # per paper: count authors, then average per year
    avg = (
        df.groupby(["paper_id", "year"])["author_id"].count()
        .reset_index(name="n_authors")
        .groupby("year")["n_authors"].mean().reset_index()
    )
    fig = px.line(
        avg, x="year", y="n_authors",
        title="Average Authors per Paper over Time",
        labels={"n_authors": "Avg authors", "year": "Year"},
        markers=True, template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig


# ---------------------------------------------------------------------------
# TITLES
# ---------------------------------------------------------------------------

def title_length_dist(data: DBData) -> go.Figure:
    """Histogram + line: title word count distribution and median per year."""
    p = data.papers.copy()
    p["title_len"] = p["title"].str.split().str.len()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Title Length Distribution (capped at 25 words)",
                        "Median Title Length per Year"),
    )
    fig.add_trace(go.Histogram(
        x=p["title_len"].clip(upper=25), nbinsx=25,
        marker_color=COLOR_SEQ[0], name="papers",
    ), row=1, col=1)

    med = p.groupby(["year", "type"])["title_len"].median().reset_index()
    for i, t in enumerate(med["type"].unique()):
        sub = med[med["type"] == t]
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["title_len"],
            mode="lines+markers", name=t,
            line=dict(color=COLOR_SEQ[i % len(COLOR_SEQ)]),
        ), row=1, col=2)

    fig.update_layout(
        template=TEMPLATE, height=420,
        title_text="Paper Title Length Analysis",
    )
    return fig


# ---------------------------------------------------------------------------
# DENSITY / MISC
# ---------------------------------------------------------------------------

def density_heatmap_year_vs_authors(data: DBData) -> go.Figure:
    """Density heatmap: year vs number of authors per paper."""
    df = (
        data.paper_authors.groupby("paper_id")["author_id"]
        .count().rename("n_authors").reset_index()
        .merge(data.papers[["id", "year"]], left_on="paper_id", right_on="id")
    )
    df = df[df["n_authors"] <= 20]
    fig = px.density_heatmap(
        df, x="year", y="n_authors",
        nbinsx=20, nbinsy=20,
        color_continuous_scale="Blues",
        title="Density Heatmap: Year vs Authors per Paper",
        labels={"n_authors": "Authors per paper", "year": "Year"},
        template=TEMPLATE,
    )
    return fig


def sunburst_year_type_venue(data: DBData, top_venues: int = 10) -> go.Figure:
    """Sunburst: year → type → top N venues."""
    top_n = (
        data.papers.dropna(subset=["venue"])
        .groupby("venue").size().nlargest(top_venues).index
    )
    df = (
        data.papers[data.papers["venue"].isin(top_n)]
        .groupby(["year", "type", "venue"]).size().reset_index(name="n_papers")
    )
    df["year"] = df["year"].astype(str)
    fig = px.sunburst(
        df, path=["year", "type", "venue"], values="n_papers",
        title=f"Sunburst: Year → Type → Top {top_venues} Venues",
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    return fig