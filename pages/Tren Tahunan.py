from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Tren Tahunan",
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
CORE_TREND_COLUMNS = ["P0", "PDRB", "IPM", "AHH"]
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

def status_perubahan_cluster(cluster_awal: int, cluster_akhir: int) -> str:
    if cluster_akhir > cluster_awal:
        return "Meningkat"
    if cluster_akhir < cluster_awal:
        return "Menurun"
    return "Tetap"

def add_cluster_description(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    result_df["Keterangan Cluster"] = result_df["Cluster"].apply(format_cluster)
    return result_df

def get_cluster_reference(df: pd.DataFrame):
    cluster_centers = (
        df
        .groupby("Cluster")[NUMERIC_COLUMNS]
        .mean()
        .sort_index()
    )

    mean_values = df[NUMERIC_COLUMNS].mean()
    std_values = df[NUMERIC_COLUMNS].std(ddof=0).replace(0, 1)

    cluster_centers_scaled = (cluster_centers - mean_values) / std_values

    return mean_values, std_values, cluster_centers_scaled

def tentukan_cluster_dari_rata_rata_indikator(
    row: pd.Series,
    mean_values: pd.Series,
    std_values: pd.Series,
    cluster_centers_scaled: pd.DataFrame,
) -> int:
    row_scaled = (row[NUMERIC_COLUMNS] - mean_values) / std_values

    distances = (
        (cluster_centers_scaled - row_scaled) ** 2
    ).sum(axis=1) ** 0.5
    return int(distances.idxmin())

try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Gagal memuat dataset: {exc}")
    st.stop()

df = add_cluster_description(df)

mean_values, std_values, cluster_centers_scaled = get_cluster_reference(df)

st.title("Tren Tahunan")
st.caption(
    "Halaman ini digunakan untuk melihat perubahan indikator ekonomi dan kesehatan "
    "masyarakat dari tahun ke tahun pada tingkat provinsi di Indonesia."
)

all_years = sorted(df["Tahun"].unique().tolist())
all_provinces = sorted(df["Provinsi"].unique().tolist())

with st.sidebar:
    st.header("Filter Tren Tahunan")
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
    )   
    selected_clusters = st.multiselect(
    "Pilih Cluster",
    options=sorted(df["Cluster"].unique().tolist()),
    default=[],
    format_func=lambda x: format_cluster(x),
    help="Pilih cluster wilayah yang ingin ditampilkan.",
    )
    selected_indicator = st.selectbox(
        "Indikator Utama",
        options=["P0", "PDRB", "IPM", "AHH", "Jumlah_Miskin"],
        index=0,
    )

filtered_df = df[
    (df["Tahun"] >= selected_years[0]) &
    (df["Tahun"] <= selected_years[1])
].copy()

if selected_provinces:
    filtered_df = filtered_df[
        filtered_df["Provinsi"].isin(selected_provinces)
    ].copy()

if selected_clusters:
    filtered_df = filtered_df[
        filtered_df["Cluster"].isin(selected_clusters)
    ].copy()

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
    st.stop()

st.markdown("### Ringkasan Tren berdasarkan Rentang Tahun Aktif")
trend_summary = filtered_df.groupby("Tahun")[NUMERIC_COLUMNS].mean().reset_index()
summary_cols = st.columns(4)
summary_cols[0].metric(
    f"P0 ({selected_years[0]} → {selected_years[1]})",
    f"{format_number(trend_summary['P0'].iloc[-1])}",
    delta=f"{format_number(trend_summary['P0'].iloc[-1] - trend_summary['P0'].iloc[0])}",
)
summary_cols[1].metric(
    f"PDRB ({selected_years[0]} → {selected_years[1]})",
    f"{format_number(trend_summary['PDRB'].iloc[-1])}",
    delta=f"{format_number(trend_summary['PDRB'].iloc[-1] - trend_summary['PDRB'].iloc[0])}",
)
summary_cols[2].metric(
    f"IPM ({selected_years[0]} → {selected_years[1]})",
    f"{format_number(trend_summary['IPM'].iloc[-1])}",
    delta=f"{format_number(trend_summary['IPM'].iloc[-1] - trend_summary['IPM'].iloc[0])}",
)
summary_cols[3].metric(
    f"AHH ({selected_years[0]} → {selected_years[1]})",
    f"{format_number(trend_summary['AHH'].iloc[-1])}",
    delta=f"{format_number(trend_summary['AHH'].iloc[-1] - trend_summary['AHH'].iloc[0])}",
)

st.markdown("### Tabel Rata-rata Indikator per Tahun")
trend_table = (
    filtered_df.groupby("Tahun", as_index=False)[NUMERIC_COLUMNS]
    .mean()
    .round(2)
    .sort_values("Tahun")
    .reset_index(drop=True)
)

trend_table["Cluster"] = trend_table.apply(
    lambda row: tentukan_cluster_dari_rata_rata_indikator(
        row,
        mean_values,
        std_values,
        cluster_centers_scaled,
    ),
    axis=1,
)

trend_table["Keterangan Cluster"] = trend_table["Cluster"].apply(format_cluster)

trend_table.index = trend_table.index + 1
trend_table = trend_table.reset_index().rename(columns={"index": "No"})

st.dataframe(trend_table, use_container_width=True, hide_index=True)

st.markdown("### Tren Indikator Utama")
if selected_provinces:
    chart_df = filtered_df.copy()
    fig_main = px.line(
        chart_df,
        x="Tahun",
        y=selected_indicator,
        color="Provinsi",
        markers=True,
        title=f"Tren {selected_indicator} pada Provinsi Terpilih ({selected_years[0]}–{selected_years[1]})",
    )
else:
    chart_df = filtered_df.groupby("Tahun", as_index=False)[selected_indicator].mean()
    fig_main = px.line(
        chart_df,
        x="Tahun",
        y=selected_indicator,
        markers=True,
        title=f"Tren Rata-rata {selected_indicator} Seluruh Provinsi ({selected_years[0]}–{selected_years[1]})",
    )
fig_main.update_layout(margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig_main, use_container_width=True)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Tren Tiap Indikator")
    multi_indicator = st.selectbox(
        "Pilih indikator untuk tren tambahan",
        options=["P0", "PDRB", "IPM", "AHH", "Jumlah_Miskin"],
        index=1,
        key="additional_indicator",
    )
    if selected_provinces:
        add_df = filtered_df.copy()
        fig_add = px.line(
            add_df,
            x="Tahun",
            y=multi_indicator,
            color="Provinsi",
            markers=True,
            title=f"Tren {multi_indicator} pada Provinsi Terpilih",
        )
    else:
        add_df = filtered_df.groupby("Tahun", as_index=False)[multi_indicator].mean()
        fig_add = px.line(
            add_df,
            x="Tahun",
            y=multi_indicator,
            markers=True,
            title=f"Tren Rata-rata {multi_indicator} Seluruh Provinsi",
        )
    fig_add.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_add, use_container_width=True)

with right_col:
    st.markdown("### Indexed Trend")
    indexed_df = filtered_df.groupby("Tahun", as_index=False)[CORE_TREND_COLUMNS].mean()
    indexed_only = indexed_df[CORE_TREND_COLUMNS].div(indexed_df[CORE_TREND_COLUMNS].iloc[0]).mul(100)
    indexed_plot_df = pd.concat([indexed_df[["Tahun"]], indexed_only], axis=1)
    indexed_long = indexed_plot_df.melt(
        id_vars="Tahun",
        value_vars=CORE_TREND_COLUMNS,
        var_name="Indikator",
        value_name="Indeks",
    )
    fig_index = px.line(
        indexed_long,
        x="Tahun",
        y="Indeks",
        color="Indikator",
        markers=True,
        title="Indexed Trend (Tahun Awal = 100)",
    )
    fig_index.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_index, use_container_width=True)

if selected_years[0] == selected_years[1]:
    st.info(
        "Bagian perubahan cluster dan perubahan indikator hanya ditampilkan "
        "jika rentang tahun terdiri dari minimal dua tahun."
    )
else:
    st.markdown(
        f"### Perubahan Cluster pada Rentang Tahun {selected_years[0]}–{selected_years[1]}"
    )

    cluster_start = (
        filtered_df[filtered_df["Tahun"] == selected_years[0]]
        [["Provinsi", "Cluster"]]
        .rename(columns={"Cluster": f"Cluster_{selected_years[0]}"})
    )

    cluster_end = (
        filtered_df[filtered_df["Tahun"] == selected_years[1]]
        [["Provinsi", "Cluster"]]
        .rename(columns={"Cluster": f"Cluster_{selected_years[1]}"})
    )

    cluster_change_df = cluster_start.merge(
        cluster_end,
        on="Provinsi",
        how="inner"
    )

    if cluster_change_df.empty:
        st.warning(
            "Data cluster pada tahun awal atau tahun akhir tidak tersedia untuk filter yang dipilih."
        )
    else:
        cluster_change_df[f"Keterangan {selected_years[0]}"] = cluster_change_df[
            f"Cluster_{selected_years[0]}"
        ].apply(format_cluster)

        cluster_change_df[f"Keterangan {selected_years[1]}"] = cluster_change_df[
            f"Cluster_{selected_years[1]}"
        ].apply(format_cluster)

        cluster_change_df["Status Perubahan"] = cluster_change_df.apply(
            lambda row: status_perubahan_cluster(
                row[f"Cluster_{selected_years[0]}"],
                row[f"Cluster_{selected_years[1]}"],
            ),
            axis=1,
        )

        status_summary = cluster_change_df["Status Perubahan"].value_counts()

        st.markdown("#### Ringkasan Perubahan Cluster")

        status_cols = st.columns(3)

        status_cols[0].metric(
            "Meningkat",
            f"{status_summary.get('Meningkat', 0):,} Provinsi"
        )

        status_cols[1].metric(
            "Tetap",
            f"{status_summary.get('Tetap', 0):,} Provinsi"
        )

        status_cols[2].metric(
            "Menurun",
            f"{status_summary.get('Menurun', 0):,} Provinsi"
        )

        cluster_change_df = cluster_change_df[
            [
                "Provinsi",
                f"Keterangan {selected_years[0]}",
                f"Keterangan {selected_years[1]}",
                "Status Perubahan",
            ]
        ].sort_values(["Status Perubahan", "Provinsi"])

        cluster_change_df = cluster_change_df.reset_index(drop=True)
        cluster_change_df.index = cluster_change_df.index + 1
        cluster_change_df = cluster_change_df.reset_index().rename(columns={"index": "No"})

        st.dataframe(
            cluster_change_df,
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        f"### Perubahan Indikator pada Rentang Tahun {selected_years[0]}–{selected_years[1]}"
    )

    start_row = trend_table.iloc[0]
    end_row = trend_table.iloc[-1]

    NEGATIVE_INDICATORS = ["Jumlah_Miskin", "P0"]
    POSITIVE_INDICATORS = ["PDRB", "IPM", "AHH"]

    def get_arah_perubahan(diff: float) -> str:
        if diff > 0:
            return "Meningkat"
        if diff < 0:
            return "Menurun"
        return "Stabil"

    def get_makna_perubahan(indicator: str, diff: float) -> str:
        if diff == 0:
            return "Stabil"

        if indicator in NEGATIVE_INDICATORS:
            return "Membaik" if diff < 0 else "Memburuk"

        if indicator in POSITIVE_INDICATORS:
            return "Membaik" if diff > 0 else "Memburuk"

        return "Perlu Ditinjau"

    change_rows = []

    for indicator in NUMERIC_COLUMNS:
        start_val = start_row[indicator]
        end_val = end_row[indicator]
        diff = end_val - start_val
        pct_change = (diff / start_val * 100) if start_val != 0 else 0

        change_rows.append({
            "Indikator": indicator,
            f"Nilai {selected_years[0]}": round(start_val, 2),
            f"Nilai {selected_years[1]}": round(end_val, 2),
            "Perubahan": round(diff, 2),
            "Persentase Perubahan": round(pct_change, 2),
            "Arah Perubahan": get_arah_perubahan(diff),
            "Makna Perubahan": get_makna_perubahan(indicator, diff),
        })

    change_df = pd.DataFrame(change_rows)
    change_df = change_df.reset_index(drop=True)
    change_df.index = change_df.index + 1
    change_df = change_df.reset_index().rename(columns={"index": "No"})

    st.dataframe(
        change_df,
        use_container_width=True,
        hide_index=True
    )