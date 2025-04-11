import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px
import re

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("umap_tunes.csv")

    # Safely parse pitch_histogram strings into lists of floats
    def parse_vector_string(s):
        return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", str(s))]

    df["pitch_histogram"] = df["pitch_histogram"].apply(parse_vector_string)
    return df


df = load_data()

# ---------- Search Bar ----------
tune_names = sorted(df["name"].unique())
selected_tune = st.selectbox("Search for a tune", tune_names)

# ---------- Nearest Neighbors ----------
def find_nearest_tunes(df, target_name, n=5):
    target_row = df[df["name"] == target_name]
    if target_row.empty:
        return pd.DataFrame()
    
    target_vec = np.vstack(target_row["pitch_histogram"].values)
    target_type = target_row["type"].values[0]
    same_type_df = df[df["type"] == target_type].copy()

    all_vecs = np.vstack(same_type_df["pitch_histogram"].values)
    distances = euclidean_distances(target_vec, all_vecs)[0]
    same_type_df["distance"] = distances
    
    result = same_type_df[same_type_df["name"] != target_name].sort_values("distance").head(n)
    return result[["name", "mode", "type", "distance"]]

nearest_df = find_nearest_tunes(df, selected_tune)

# ---------- Plot ----------
fig = px.scatter(
    df, x="x", y="y", color="mode", hover_data=["name", "type"],
    title="UMAP of Irish Tune Pitch Histograms", opacity=0.5
)

# Highlight selected tune
selected_row = df[df["name"] == selected_tune]
if not selected_row.empty:
    fig.add_scatter(
        x=selected_row["x"],
        y=selected_row["y"],
        mode="markers+text",
        marker=dict(size=12, color="black", symbol="x"),
        text=["🎯"],
        name="Selected Tune"
    )

st.plotly_chart(fig, use_container_width=True)

# ---------- Show Nearest ----------
st.subheader("Closest Tunes (same type):")
st.dataframe(nearest_df.reset_index(drop=True), use_container_width=True)
