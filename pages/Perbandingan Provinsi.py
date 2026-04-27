from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Perbandingan Provinsi",
    layout="wide"
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
]
NUMERIC_COLUMNS = ["Jumlah_Miskin", "P0", "PDRB", "IPM", "AHH"]
RANKING_INDICATORS = ["P0", "PDRB", "IPM", "AHH", "Jumlah_Miskin"]
HEATMAP_INDICATORS = ["P0", "IPM", "AHH", "PDRB"]


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
    for col in ["Tahun", *NUMERIC_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values(["Provinsi", "Tahun"]).reset_index(drop=True)
    return df


def format_number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Gagal memuat dataset: {exc}")
    st.stop()

st.title("Perbandingan Antar Provinsi")
st.caption(
    "Halaman ini digunakan untuk membandingkan kondisi ekonomi dan kesehatan masyarakat "
    "antar provinsi berdasarkan tahun atau rentang tahun yang dipilih."
)

all_years = sorted(df["Tahun"].unique().tolist())
all_provinces = sorted(df["Provinsi"].unique().tolist())
latest_year = int(max(all_years))

with st.sidebar:
    st.header("Filter Perbandingan Provinsi")
    selected_years = st.slider(
        "Rentang Tahun",
        min_value=int(min(all_years)),
        max_value=int(max(all_years)),
        value=(int(min(all_years)), int(max(all_years))),
    )

    sort_order = st.radio(
        "Urutan Ranking",
        options=["Tertinggi", "Terendah"],
        index=0,
        horizontal=True,
    )
    top_n = st.slider("Jumlah Provinsi", min_value=5, max_value=34, value=10)


    selected_compare_provinces = st.multiselect(
        "Provinsi untuk Tren Perbandingan",
        options=all_provinces,
    )

filtered_df = df[
    (df["Tahun"] >= selected_years[0]) &
    (df["Tahun"] <= selected_years[1])
].copy()

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
    st.stop()

st.markdown("### Ringkasan Wilayah")
summary_cols = st.columns(4)
summary_cols[0].metric("Jumlah Provinsi", f"{filtered_df['Provinsi'].nunique():,}")
summary_cols[1].metric("Tahun Aktif", f"{selected_years[0]} - {selected_years[1]}")

st.markdown("### Ranking Provinsi")

selected_indicator = st.selectbox(
    "Pilih Indikator Ranking",
    options=RANKING_INDICATORS,
    index=0,
    key="indicator_rank_compare"
)

ranking_base = filtered_df.groupby("Provinsi", as_index=False)[RANKING_INDICATORS].mean()
ranking_title_period = f"rata-rata {selected_years[0]}–{selected_years[1]}"

ascending = True if sort_order == "Terendah" else False

ranking_df = ranking_base.sort_values(
    selected_indicator, ascending=ascending
).head(top_n)

col_rank_chart, col_rank_table = st.columns([2, 1])

with col_rank_chart:
    fig_rank = px.bar(
        ranking_df,
        x=selected_indicator,
        y="Provinsi",
        orientation="h",
        title=f"{top_n} Provinsi {sort_order} berdasarkan {selected_indicator} ({ranking_title_period})",
    )
    fig_rank.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(categoryorder="total ascending"),
    )
    st.plotly_chart(fig_rank, use_container_width=True)

with col_rank_table:
    st.markdown("#### Tabel Ranking")
    ranking_table = ranking_df[["Provinsi", selected_indicator]].reset_index(drop=True)
    ranking_table.index = ranking_table.index + 1
    ranking_table =ranking_table.reset_index().rename(columns={"index": "No"})
    st.dataframe(ranking_table, use_container_width=True,hide_index=True)

st.markdown("### Heatmap Provinsi dan Tahun")

heatmap_indicator = st.selectbox(
    "Pilih Indikator Heatmap",
    options=HEATMAP_INDICATORS,
    index=0,
    key="heatmap_indicator"
)

heatmap_source = (
    filtered_df.groupby(["Provinsi", "Tahun"], as_index=False)[heatmap_indicator]
    .mean()
)

heatmap_pivot = heatmap_source.pivot(
    index="Provinsi",
    columns="Tahun",
    values=heatmap_indicator
)

fig_heatmap = px.imshow(
    heatmap_pivot,
    aspect="auto",
    color_continuous_scale="RdYlGn_r",
    labels=dict(x="Tahun", y="Provinsi", color=heatmap_indicator),
    title=f"Heatmap {heatmap_indicator} per Provinsi dan Tahun",
)

fig_heatmap.update_layout(margin=dict(l=20, r=20, t=50, b=20))

st.plotly_chart(fig_heatmap, use_container_width=True)
st.markdown("### Perbandingan Tren Provinsi Terpilih")
if not selected_compare_provinces:
    st.info("Pilih minimal satu provinsi pada sidebar untuk menampilkan tren perbandingan.")
else:
    compare_df = filtered_df[filtered_df["Provinsi"].isin(selected_compare_provinces)].copy()
    trend_indicator = st.selectbox(
        "Pilih indikator tren provinsi",
        options=RANKING_INDICATORS,
        index=0,
    )
    compare_grouped = compare_df.groupby(["Provinsi", "Tahun"], as_index=False)[trend_indicator].mean()
    fig_compare = px.line(
        compare_grouped,
        x="Tahun",
        y=trend_indicator,
        color="Provinsi",
        markers=True,
        title=f"Tren {trend_indicator} pada Provinsi Terpilih",
    )
    fig_compare.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("### Snapshot Tahun Terbaru")

latest_snapshot = filtered_df[filtered_df["Tahun"] == filtered_df["Tahun"].max()].copy()

latest_snapshot = (
    latest_snapshot[["Provinsi", "Tahun", "P0", "PDRB", "IPM", "AHH", "Jumlah_Miskin"]]
    .sort_values("Provinsi")
    .reset_index(drop=True)
)
latest_snapshot.index = latest_snapshot.index + 1
latest_snapshot = latest_snapshot.reset_index().rename(columns={"index": "No"})

st.dataframe(latest_snapshot, use_container_width=True, hide_index=True)

