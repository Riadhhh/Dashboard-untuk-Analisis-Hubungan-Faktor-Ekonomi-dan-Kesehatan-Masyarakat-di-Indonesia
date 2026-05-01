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
]
NUMERIC_COLUMNS = ["Jumlah_Miskin", "P0", "PDRB", "IPM", "AHH"]
CORE_TREND_COLUMNS = ["P0", "PDRB", "IPM", "AHH"]


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
    filtered_df = filtered_df[filtered_df["Provinsi"].isin(selected_provinces)].copy()
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

st.markdown("### Tabel Tren Tahunan")
trend_table = (
    filtered_df.groupby("Tahun", as_index=False)[NUMERIC_COLUMNS]
    .mean()
    .round(2)
    .sort_values("Tahun")
    .reset_index(drop=True)
)

trend_table.index = trend_table.index + 1
trend_table = trend_table.reset_index().rename(columns={"index": "No"})

st.dataframe(trend_table, use_container_width=True, hide_index=True)

st.markdown(f"### Perubahan Indikator pada Rentang Tahun {selected_years[0]}–{selected_years[1]}")
start_row = trend_table.iloc[0]
end_row = trend_table.iloc[-1]
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
        "Status": "Meningkat" if diff > 0 else "Menurun" if diff < 0 else "Stabil",
    })
change_df = pd.DataFrame(change_rows)
change_df = change_df.reset_index(drop=True)
change_df.index = change_df.index + 1
change_df = change_df.reset_index().rename(columns={"index": "No"})

st.dataframe(change_df, use_container_width=True, hide_index=True)