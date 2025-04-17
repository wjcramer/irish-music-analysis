import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px
import plotly.graph_objects as go
import re
import json

# ---------- Load Data ----------
@st.cache_data
def load_data():
    # Load main UMAP data
    df = pd.read_csv("umap_tunes.csv")

    # Parse pitch histograms
    def parse_vector_string(s):
        return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", str(s))]

    df["pitch_histogram"] = df["pitch_histogram"].apply(parse_vector_string)
    df = df[df["pitch_histogram"].apply(lambda x: isinstance(x, list) and len(x) == 12)].copy()

    # Load and merge popularity data
    with open("tune_popularity.json", "r") as f:
        popularity_data = json.load(f)

    df_pop = pd.DataFrame(popularity_data)
    df_pop["tunebooks"] = pd.to_numeric(df_pop["tunebooks"], errors="coerce")
    df = df.merge(df_pop[["name", "tunebooks"]], on="name", how="left")

    # Fill NaN with 0 if some tunes aren’t in the popularity list
    df["tunebooks"] = df["tunebooks"].fillna(0).astype(int)

    return df


df = load_data()

# ---------- Search Bar ----------
tune_names = sorted(df["name"].unique())

st.markdown("🎻 **Search for a tune by name**")

selected_tune = st.selectbox(
    "",
    options=[""] + sorted(df["name"].unique()),  # prepend empty option
    index=0,
    placeholder="Start typing a tune name..."
)

# ---------- Popularity check box ----------
popular_only = st.checkbox("🔍 Only include more popular tunes (added to ≥ 15 tunebooks)", value=False)


# ---------- Nearest Neighbors ----------
def find_nearest_tunes(df, target_name, popular_only=False, n=5):
    target_row = df[df["name"] == target_name]
    if target_row.empty:
        return pd.DataFrame()

    target_vec = np.vstack(target_row["pitch_histogram"].values)
    target_type = target_row["type"].values[0]
    same_type_df = df[df["type"] == target_type].copy()

    if popular_only:
        same_type_df = same_type_df[same_type_df["tunebooks"] >= 15]

    all_vecs = np.vstack(same_type_df["pitch_histogram"].values)
    distances = euclidean_distances(target_vec, all_vecs)[0]
    same_type_df["distance"] = distances

    result = same_type_df[same_type_df["name"] != target_name].sort_values("distance").head(n)
    return result[["name", "mode", "type", "tunebooks", "distance"]]


# ---------- UMAP Plot (always visible) ----------
fig = go.Figure()

# Base scatter plot (all tunes except selected one)
for mode in df["mode"].unique():
    subset = df[(df["mode"] == mode) & (df["name"] != selected_tune)]
    fig.add_trace(go.Scatter(
        x=subset["x"],
        y=subset["y"],
        mode="markers",
        name=mode,
        marker=dict(size=5, opacity=0.5),
        text=subset["name"],
        hoverinfo="text"
    ))

# ---------- Add selected tune as black X if selected ----------
if selected_tune:
    selected_row = df[df["name"] == selected_tune]
    if not selected_row.empty:
        fig.add_trace(go.Scatter(
            x=selected_row["x"],
            y=selected_row["y"],
            mode="markers",
            marker=dict(size=14, color="black", symbol="x"),
            name=selected_tune,
            showlegend=True,
            hoverinfo="skip"
        ))

# ---------- Layout ----------
fig.update_layout(
    title="UMAP of Irish Tune Pitch Histograms",
    xaxis_title="x",
    yaxis_title="y",
    legend_title="Mode",
    margin=dict(l=20, r=20, t=40, b=20),
    legend_traceorder="normal"  # or "reversed" if you want selected on top
)

st.plotly_chart(fig, use_container_width=True)


# Only show nearest tunes table *after* tune is selected
if selected_tune:
    nearest_df = find_nearest_tunes(df, selected_tune, popular_only=popular_only)

    # Rename columns for better display
    nearest_df = nearest_df.rename(columns={
        "tunebooks": "Popularity (tunebooks)",
        "distance": "Degree of similarity (smaller is more similar)"
    })

    st.subheader("Closest Tunes (same type):")
    st.dataframe(nearest_df.reset_index(drop=True), use_container_width=True)

