import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Dashboard Ekonomi dan Kesehatan Indonesia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
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

    numeric_cols = ["Tahun", "Jumlah_Miskin", "P0", "PDRB", "IPM", "AHH"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values(["Provinsi", "Tahun"]).reset_index(drop=True)
    return df


def format_number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def add_number_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return df.reset_index().rename(columns={"index": "No"})


st.title("Dashboard Faktor Ekonomi dan Kesehatan Masyarakat di Indonesia")
st.caption(
    "Pengembangan dashboard berbasis Streamlit untuk mengeksplorasi hubungan faktor ekonomi "
    "dan kesehatan masyarakat pada tingkat provinsi periode 2015–2024."
)

try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Gagal memuat dataset: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("**Sumber data:** Badan Pusat Statistik Indonesia (BPS)")
    st.markdown("**Unit analisis:** Provinsi per tahun")
    st.markdown("**Periode:** 2015–2024")

st.subheader("Ringkasan Dataset")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Baris", f"{len(df):,}")
col2.metric("Jumlah Provinsi", f"{df['Provinsi'].nunique():,}")
col3.metric("Tahun Awal", int(df["Tahun"].min()))
col4.metric("Tahun Akhir", int(df["Tahun"].max()))

st.markdown("### Rata-rata Indikator Seluruh Periode")
mean_cols = st.columns(4)
mean_cols[0].metric("Rata-rata P0", format_number(df["P0"].mean()))
mean_cols[1].metric("Rata-rata PDRB", format_number(df["PDRB"].mean()))
mean_cols[2].metric("Rata-rata IPM", format_number(df["IPM"].mean()))
mean_cols[3].metric("Rata-rata AHH", format_number(df["AHH"].mean()))

st.markdown("### Tujuan Dashboard")
st.info(
    "Dashboard ini dirancang untuk menampilkan kondisi umum indikator ekonomi dan kesehatan, "
    "melihat tren tahunan, membandingkan antar provinsi, serta menganalisis hubungan "
    "antara faktor ekonomi dan kesehatan masyarakat di Indonesia."
)

st.markdown("### Variabel yang Digunakan dan Definisinya")

var1, var2 = st.columns(2)

with var1:
    st.markdown(
        """
        **1. Jumlah_Miskin**  
        Penduduk yang memiliki rata-rata pengeluaran per kapita per bulan di bawah garis kemiskinan.

        **2. P0 (Persentase Penduduk Miskin)**  
        Persentase penduduk yang memiliki rata-rata pengeluaran per kapita per bulan di bawah garis kemiskinan.

        **3. PDRB (Produk Domestik Regional Bruto)**  
        Nilai produk atau barang dan jasa yang dihasilkan di dalam wilayah domestik untuk digunakan sebagai konsumsi akhir masyarakat.
        """
    )

with var2:
    st.markdown(
        """
        **4. IPM (Indeks Pembangunan Manusia)**  
        Indeks yang mengukur pembangunan manusia dari tiga aspek dasar, yaitu umur panjang dan hidup sehat, pengetahuan, dan standar hidup layak.

        **5. AHH (Angka Harapan Hidup)**  
        Rata-rata perkiraan tahun yang dapat dijalani seseorang sejak lahir.

        **Provinsi dan Tahun**  
        Digunakan sebagai identitas unit analisis data pada tingkat provinsi per tahun.
        """
    )

st.markdown("### Pratinjau Data")
preview_df = add_number_column(df.head(340))
st.dataframe(preview_df, use_container_width=True, hide_index=True)

st.markdown("### Kesimpulan Hubungan Faktor Ekonomi dan Kesehatan")

kesimpulan_df = pd.DataFrame(
    [
        {
            "Fokus Hubungan": "P0 dengan AHH",
            "Kesimpulan": "Kemiskinan yang lebih tinggi cenderung berkaitan dengan angka harapan hidup yang lebih rendah.",
        },
        {
            "Fokus Hubungan": "P0 dengan IPM",
            "Kesimpulan": "Kemiskinan yang lebih tinggi cenderung berkaitan dengan kualitas pembangunan manusia yang lebih rendah.",
        },
        {
            "Fokus Hubungan": "PDRB dengan AHH",
            "Kesimpulan": "Skala ekonomi wilayah yang lebih besar cenderung berkaitan dengan angka harapan hidup yang lebih tinggi.",
        },
        {
            "Fokus Hubungan": "PDRB dengan IPM",
            "Kesimpulan": "Skala ekonomi wilayah yang lebih besar cenderung berkaitan dengan IPM yang lebih tinggi.",
        },
    ]
)

kesimpulan_df = add_number_column(kesimpulan_df)
st.dataframe(kesimpulan_df, use_container_width=True, hide_index=True)

st.success(
    "Secara umum, hasil analisis menunjukkan bahwa indikator kemiskinan cenderung berhubungan "
    "negatif dengan indikator kesehatan dan pembangunan manusia, sedangkan PDRB cenderung "
    "berhubungan positif dengan AHH dan IPM. Dengan demikian, dashboard ini mendukung analisis "
    "hubungan antara faktor ekonomi dan kesehatan masyarakat di Indonesia."
)

st.markdown("### Arah Eksplorasi")
nav1, nav2, nav3, nav4 = st.columns(4)
nav1.success("Overview\n\nRingkasan indikator dan kondisi umum data")
nav2.info("Tren Tahunan\n\nPerubahan indikator dari 2015–2024")
nav3.warning("Perbandingan Provinsi\n\nRanking dan persebaran antar wilayah")
nav4.error("Hubungan Variabel\n\nScatterplot dan korelasi Pearson")