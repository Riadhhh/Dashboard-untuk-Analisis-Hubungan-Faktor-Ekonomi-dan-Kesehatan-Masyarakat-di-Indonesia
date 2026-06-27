from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Overview",
    layout="wide",
    page_icon=None
    )

DATA_PATH = Path("data/dataset_final.csv")
EXPECTED_COLUMNS = [
    "Provinsi",
    "Tahun",
    "Jumlah_Miskin",
    "P0",
    "PDRB",
    "IPM",
    "AHH",
    "Cluster",
]
NUMERIC_COLUMNS = ["Jumlah_Miskin", "P0", "PDRB", "IPM", "AHH"]
CLUSTER_LABELS = {
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File dataset tidak ditemukan di: {path}. "
            "Pastikan dataset_final.csv berada di folder data/."
        )

    df = pd.read_csv(path)
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Kolom dataset belum lengkap. Kolom yang belum ditemukan: "
            + ", ".join(missing_cols)
        )

    df = df[EXPECTED_COLUMNS].copy()

    for col in ["Tahun", *NUMERIC_COLUMNS, "Cluster"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values(["Provinsi", "Tahun"]).reset_index(drop=True)

    df["Tahun"] = df["Tahun"].astype(int)
    df["Cluster"] = df["Cluster"].astype(int)

    return df

def format_number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"

def format_cluster(value: int) -> str:
    return CLUSTER_LABELS.get(int(value), "Tidak Diketahui")

try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Gagal memuat dataset: {exc}")
    st.stop()

st.title("Overview Dashboard")
st.caption(
    "Halaman ini menampilkan gambaran umum data ekonomi dan kesehatan masyarakat "
    "pada tingkat provinsi di Indonesia periode 2015–2024."
)

all_years = sorted(df["Tahun"].unique().tolist())
all_provinces = sorted(df["Provinsi"].unique().tolist())
latest_year = max(all_years)

with st.sidebar:
    st.header("Filter Overview")
    selected_years = st.slider(
        "Rentang Tahun",
        min_value=int(min(all_years)),
        max_value=int(max(all_years)),
        value=(int(min(all_years)), int(max(all_years))),
    )
    selected_provinces = st.multiselect(
        "Pilih Provinsi",
        options=all_provinces,
        default=[],
        help="Kosongkan pilihan untuk menampilkan seluruh provinsi.",
    )

    top_n = st.slider("Jumlah Provinsi", min_value=5, max_value=34, value=10)

filtered_df = df[
    (df["Tahun"] >= selected_years[0]) &
    (df["Tahun"] <= selected_years[1])
].copy()

if selected_provinces:
    filtered_df = filtered_df[filtered_df["Provinsi"].isin(selected_provinces)].copy()

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
    st.stop()

metric_cols = st.columns(5)
metric_cols[0].metric("Jumlah Baris", f"{len(filtered_df):,}")
metric_cols[1].metric("Jumlah Provinsi", f"{filtered_df['Provinsi'].nunique():,}")
metric_cols[2].metric("Rata-rata P0", format_number(filtered_df["P0"].mean()))
metric_cols[3].metric("Rata-rata IPM", format_number(filtered_df["IPM"].mean()))
metric_cols[4].metric("Rata-rata AHH", format_number(filtered_df["AHH"].mean()))

metric_cols_2 = st.columns(3)
metric_cols_2[0].metric("Rata-rata PDRB", format_number(filtered_df["PDRB"].mean()))
metric_cols_2[1].metric("Rata-rata Jumlah_Miskin", format_number(filtered_df["Jumlah_Miskin"].mean()))
metric_cols_2[2].metric("Rentang Tahun Aktif", f"{selected_years[0]} - {selected_years[1]}")

latest_active_year = int(filtered_df["Tahun"].max())

latest_cluster_df = filtered_df[
    filtered_df["Tahun"] == latest_active_year
].copy()

cluster_counts = (
    latest_cluster_df
    .groupby("Cluster")["Provinsi"]
    .nunique()
    .reindex([1, 2, 3], fill_value=0)
)

st.markdown(f"### Ringkasan Cluster Tahun Terbaru ({latest_active_year})")

cluster_cols = st.columns(3)

cluster_cols[0].metric(
    "Cluster 1",
    f"{cluster_counts.loc[1]:,} Provinsi"
)

cluster_cols[1].metric(
    "Cluster 2",
    f"{cluster_counts.loc[2]:,} Provinsi"
)

cluster_cols[2].metric(
    "Cluster 3",
    f"{cluster_counts.loc[3]:,} Provinsi"
)

st.markdown("### Ringkasan Statistik")
summary_df = pd.DataFrame({
    "Rata-rata": filtered_df[NUMERIC_COLUMNS].mean(),
    "Median": filtered_df[NUMERIC_COLUMNS].median(),
    "Minimum": filtered_df[NUMERIC_COLUMNS].min(),
    "Maksimum": filtered_df[NUMERIC_COLUMNS].max(),
}).round(2)
summary_df.index.name = "Indikator"
st.dataframe(summary_df, use_container_width=True)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Tren Rata-rata Indikator")
    trend_df = filtered_df.groupby("Tahun")[NUMERIC_COLUMNS].mean().reset_index()
    trend_indicator = st.selectbox(
        "Pilih indikator tren",
        options=["P0", "PDRB", "IPM", "AHH", "Jumlah_Miskin"],
        index=0,
        key="overview_trend_indicator",
    )
    fig_trend = px.line(
        trend_df,
        x="Tahun",
        y=trend_indicator,
        markers=True,
        title=f"Tren Rata-rata {trend_indicator}",
    )
    fig_trend.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_trend, use_container_width=True)

with right_col:
    st.markdown("### Ranking Provinsi berdasarkan Rentang Tahun")
    selected_indicator = st.selectbox(
        "Pilih Indikator Ranking",
        options=["P0", "PDRB", "IPM", "AHH", "Jumlah_Miskin"],
        index=0,
        key="indicator_rank"
    )
    ranking_df = (
        filtered_df.groupby("Provinsi", as_index=False)[selected_indicator]
        .mean()
        .sort_values(selected_indicator, ascending=False)
        .head(top_n)
    )
    fig_rank = px.bar(
        ranking_df,
        x=selected_indicator,
        y="Provinsi",
        orientation="h",
        title=(
            f"Top {top_n} Provinsi berdasarkan rata-rata {selected_indicator} "
            f"({selected_years[0]}–{selected_years[1]})"
        ),
    )
    fig_rank.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(categoryorder="total ascending")
    )
    st.plotly_chart(fig_rank, use_container_width=True)

st.markdown("### Pratinjau Tahun Terbaru")

latest_snapshot = filtered_df[
    filtered_df["Tahun"] == filtered_df["Tahun"].max()
].copy()

latest_snapshot["Keterangan Cluster"] = latest_snapshot["Cluster"].apply(format_cluster)

latest_snapshot = (
    latest_snapshot[
        [
            "Provinsi",
            "Tahun",
            "P0",
            "PDRB",
            "IPM",
            "AHH",
            "Jumlah_Miskin",
            "Cluster",
            "Keterangan Cluster",
        ]
    ]
    .sort_values("Provinsi")
    .reset_index(drop=True)
)

latest_snapshot.index = latest_snapshot.index + 1
latest_snapshot = latest_snapshot.reset_index().rename(columns={"index": "No"})

st.dataframe(
    latest_snapshot,
    use_container_width=True,
    hide_index=True
)

