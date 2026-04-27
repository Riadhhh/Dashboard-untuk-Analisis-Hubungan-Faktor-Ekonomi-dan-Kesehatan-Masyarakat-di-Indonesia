from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import pearsonr

st.set_page_config(
    page_title="Hubungan Ekonomi dan Kesehatan", 
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
MAIN_PAIRS = [("P0", "AHH"), ("PDRB", "AHH"), ("P0", "IPM"), ("PDRB", "IPM")]
SUPPORTING_PAIRS = [("Jumlah_Miskin", "AHH"), ("Jumlah_Miskin", "IPM")]
ALL_PAIRS = MAIN_PAIRS + SUPPORTING_PAIRS
PAIR_OPTIONS = [f"{x} vs {y}" for x, y in ALL_PAIRS]


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


def format_number(value: float, digits: int = 4) -> str:
    return f"{value:,.{digits}f}"


def arah_hubungan(r: float) -> str:
    if r > 0:
        return "Positif"
    if r < 0:
        return "Negatif"
    return "Tidak ada hubungan linear"


def kekuatan_hubungan(r: float) -> str:
    abs_r = abs(r)
    if abs_r < 0.20:
        return "Sangat lemah"
    if abs_r < 0.40:
        return "Lemah"
    if abs_r < 0.60:
        return "Sedang"
    if abs_r < 0.80:
        return "Kuat"
    return "Sangat kuat"


def signifikansi_p(p: float) -> str:
    return "Signifikan" if p < 0.05 else "Tidak signifikan"


def interpretasi_singkat(x: str, y: str) -> str:
    if x == "P0" and y == "AHH":
        return "Semakin tinggi kemiskinan, cenderung semakin rendah angka harapan hidup."
    if x == "PDRB" and y == "AHH":
        return "Semakin tinggi PDRB, cenderung semakin tinggi angka harapan hidup."
    if x == "P0" and y == "IPM":
        return "Semakin tinggi kemiskinan, cenderung semakin rendah IPM."
    if x == "PDRB" and y == "IPM":
        return "Semakin tinggi PDRB, cenderung semakin tinggi IPM."
    if x == "Jumlah_Miskin" and y == "AHH":
        return "Hubungan perlu dibaca hati-hati karena Jumlah_Miskin bersifat absolut."
    if x == "Jumlah_Miskin" and y == "IPM":
        return "Hubungan perlu dibaca hati-hati karena Jumlah_Miskin bersifat absolut."
    return "Hubungan antar variabel perlu diinterpretasikan sesuai konteks penelitian."


def add_number_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return df.reset_index().rename(columns={"index": "No"})


def build_pearson_table(data: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for x, y in pairs:
        r, p = pearsonr(data[x], data[y])
        rows.append(
            {
                "Variabel X": x,
                "Variabel Y": y,
                "Nilai r": f"{r:.4f}",
                "Arah Hubungan": arah_hubungan(r),
                "Kekuatan Hubungan": kekuatan_hubungan(r),
                "p-value": f"{p:.2f}",
                "Signifikansi": signifikansi_p(p),
                "Interpretasi Singkat": interpretasi_singkat(x, y),
            }
        )
    return add_number_column(pd.DataFrame(rows))


try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Gagal memuat dataset: {exc}")
    st.stop()

st.title("Hubungan Ekonomi dan Kesehatan")
st.caption(
    "Halaman ini menampilkan pola hubungan antara faktor ekonomi dan indikator kesehatan/pembangunan manusia, "
    "serta hasil korelasi Pearson berdasarkan data provinsi per tahun."
)

all_years = sorted(df["Tahun"].unique().tolist())
all_provinces = sorted(df["Provinsi"].unique().tolist())

with st.sidebar:
    st.header("Filter Hubungan Variabel")
    selected_years = st.slider(
        "Rentang Tahun",
        min_value=int(min(all_years)),
        max_value=int(max(all_years)),
        value=(int(min(all_years)), int(max(all_years))),
    )
    selected_provinces = st.multiselect(
        "Pilih Provinsi (opsional)",
        options=all_provinces,
        default=[],
        help="Kosongkan untuk menampilkan seluruh provinsi.",
    )
    selected_pair_label = st.selectbox(
        "Pasangan Variabel Utama",
        options=PAIR_OPTIONS,
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

selected_x, selected_y = selected_pair_label.split(" vs ")
selected_r, selected_p = pearsonr(filtered_df[selected_x], filtered_df[selected_y])

summary_cols = st.columns(4)
summary_cols[0].metric("Jumlah Observasi", f"{len(filtered_df):,}")
summary_cols[1].metric("Nilai r", format_number(selected_r))
summary_cols[2].metric("Arah Hubungan", arah_hubungan(selected_r))
summary_cols[3].metric("Signifikansi", signifikansi_p(selected_p))

st.markdown("### Scatterplot Hubungan Variabel")
scatter_df = filtered_df[["Provinsi", "Tahun", selected_x, selected_y]].copy()
fig_scatter = px.scatter(
    scatter_df,
    x=selected_x,
    y=selected_y,
    color="Provinsi" if selected_provinces else None,
    hover_data=["Provinsi", "Tahun"],
    title=f"Hubungan {selected_x} dan {selected_y} ({selected_years[0]}–{selected_years[1]})",
    opacity=0.75,
)

if scatter_df[selected_x].nunique() > 1:
    coeffs = np.polyfit(scatter_df[selected_x], scatter_df[selected_y], 1)
    x_line = np.linspace(scatter_df[selected_x].min(), scatter_df[selected_x].max(), 100)
    y_line = coeffs[0] * x_line + coeffs[1]
    fig_scatter.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Garis Tren",
            line=dict(width=3),
        )
    )

fig_scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("### Interpretasi Pasangan Variabel Terpilih")
interp_cols = st.columns(3)
interp_cols[0].metric("Kekuatan Hubungan", kekuatan_hubungan(selected_r))
interp_cols[1].metric("p-value", format_number(selected_p, digits=2))
interp_cols[2].metric("Rentang Tahun", f"{selected_years[0]} - {selected_years[1]}")
st.info(interpretasi_singkat(selected_x, selected_y))

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Heatmap Korelasi")
    corr_matrix = filtered_df[NUMERIC_COLUMNS].corr(method="pearson")
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Heatmap Korelasi Pearson",
    )
    fig_heatmap.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_heatmap, use_container_width=True)

with right_col:
    st.markdown("### Snapshot Data Analisis")
    snapshot_df = filtered_df[["Provinsi", "Tahun", "P0", "PDRB", "IPM", "AHH", "Jumlah_Miskin"]].copy()
    snapshot_df = snapshot_df.sort_values(["Provinsi", "Tahun"]).reset_index(drop=True)
    snapshot_df = add_number_column(snapshot_df.head(340))
    st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

st.markdown("### Hasil Korelasi Pearson Utama")
pearson_main_df = build_pearson_table(filtered_df, MAIN_PAIRS)
st.dataframe(pearson_main_df, use_container_width=True, hide_index=True)

st.markdown("### Hasil Korelasi Pearson Pendukung")
pearson_support_df = build_pearson_table(filtered_df, SUPPORTING_PAIRS)
st.dataframe(pearson_support_df, use_container_width=True, hide_index=True)

st.markdown("### Interpretasi Hasil Korelasi")

st.markdown("#### Hasil Utama")
for _, row in pearson_main_df.iterrows():
    st.write(
        f"- Hubungan antara **{row['Variabel X']}** dan **{row['Variabel Y']}** bersifat "
        f"**{row['Arah Hubungan'].lower()}** dengan kekuatan **{row['Kekuatan Hubungan'].lower()}** "
        f"(r = {row['Nilai r']}, p-value = {row['p-value']})."
    )

st.markdown("#### Hasil Pendukung")
for _, row in pearson_support_df.iterrows():
    st.write(
        f"- Hubungan antara **{row['Variabel X']}** dan **{row['Variabel Y']}** bersifat "
        f"**{row['Arah Hubungan'].lower()}** dengan kekuatan **{row['Kekuatan Hubungan'].lower()}** "
        f"(r = {row['Nilai r']}, p-value = {row['p-value']})."
    )

st.info(
    "Korelasi Pearson menunjukkan arah, kekuatan, dan signifikansi hubungan linear antar variabel, "
    "tetapi tidak membuktikan sebab-akibat."
)
