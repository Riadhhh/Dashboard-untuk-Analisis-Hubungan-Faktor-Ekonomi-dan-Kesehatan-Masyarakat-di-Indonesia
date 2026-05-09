import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

def get_api_key() -> str | None:
    api_key = os.getenv("API_KEY")
    if not api_key:
        api_key = st.secrets.get("API_KEY", None)
    return api_key

def generate_ai(
    pearson_main_df: pd.DataFrame,
    pearson_support_df: pd.DataFrame,
    tahun_awal: int,
    tahun_akhir: int,
    jumlah_observasi: int,
    provinsi_terpilih: list[str] | None = None,
) -> str:
    api_key = get_api_key()

    if not api_key:
        return "Kesimpulan otomatis belum dapat dibuat karena API key belum ditemukan."

    if provinsi_terpilih:
        konteks_provinsi = ", ".join(provinsi_terpilih)
    else:
        konteks_provinsi = "seluruh provinsi"

    hasil_utama = pearson_main_df.to_string(index=False)
    hasil_pendukung = pearson_support_df.to_string(index=False)

    prompt = f"""
Kamu adalah asisten analisis data pada dashboard hubungan faktor ekonomi dan kesehatan masyarakat di Indonesia.

Buat interpretasi sangat singkat dari hasil korelasi Pearson yang ditampilkan pada dashboard.

Konteks data:
- Wilayah analisis: {konteks_provinsi}
- Rentang tahun: {tahun_awal} sampai {tahun_akhir}
- Jumlah observasi: {jumlah_observasi}

Tabel hasil korelasi utama:
{hasil_utama}

Tabel hasil korelasi pendukung:
{hasil_pendukung}

Instruksi jawaban:
- Jawab dalam bahasa Indonesia.
- Buat interpretasi sangat singkat.
- Jangan membuat paragraf panjang.
- Jangan membuat pembahasan rinci.
- Jangan menyebut seluruh angka r dan p-value kecuali diperlukan.
- Gunakan maksimal 6 bullet point, untuk korelasi utama dan pendukung.
- Setiap bullet point maksimal 1 kalimat pendek.
- Fokus pada makna umum hubungan antar variabel.
- Gunakan kata "cenderung", "berkaitan", atau "menunjukkan hubungan".
- Jangan menyatakan hubungan sebab-akibat.
- Untuk Jumlah_Miskin, cukup jelaskan bahwa interpretasinya perlu hati-hati karena bersifat absolut.
- Akhiri dengan 1 kalimat singkat bahwa korelasi tidak menunjukkan sebab-akibat.

Contoh gaya jawaban:
- Semakin tinggi kemiskinan, cenderung semakin rendah angka harapan hidup.
- Semakin tinggi kemiskinan, cenderung semakin rendah IPM.
- Semakin tinggi PDRB, cenderung semakin tinggi angka harapan hidup dan IPM.
"""

    try:
        client = genai.Client(api_key=api_key)

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )

                if response.text:
                    return response.text.strip()

                return "Kesimpulan otomatis belum dapat dibuat karena respons AI kosong."

            except Exception as error:
                error_text = str(error)

                if "429" in error_text or "quota" in error_text.lower() or "rate" in error_text.lower():
                    return (
                        "Kesimpulan otomatis belum dapat dibuat karena batas penggunaan API kemungkinan telah tercapai. "
                        "Silakan tunggu beberapa saat atau kurangi jumlah permintaan."
                    )

                if "503" in error_text or "UNAVAILABLE" in error_text:
                    if attempt < 2:
                        time.sleep(2)
                        continue

                    return (
                        "Kesimpulan otomatis belum dapat dibuat karena layanan AI sedang sibuk. "
                        "Silakan coba kembali beberapa saat lagi."
                    )

                if "API_KEY" in error_text or "api key" in error_text.lower() or "permission" in error_text.lower():
                    return (
                        "Kesimpulan otomatis belum dapat dibuat karena API key tidak valid atau belum memiliki izin akses."
                    )

                if "model" in error_text.lower() or "not found" in error_text.lower():
                    return (
                        "Kesimpulan otomatis belum dapat dibuat karena model AI yang digunakan tidak tersedia."
                    )

                return f"Kesimpulan otomatis belum dapat dibuat. Detail error: {error_text}"

    except Exception:
        return "Kesimpulan otomatis belum dapat dibuat karena terjadi kendala pada proses pemanggilan AI."


def chatbot(
    user_question: str,
    data: pd.DataFrame,
    x: str,
    y: str,
    r: float,
    p: float,
    pearson_main_df: pd.DataFrame,
    pearson_support_df: pd.DataFrame,
    tahun_awal: int,
    tahun_akhir: int,
    provinsi_terpilih: list[str] | None = None,
) -> str:
    api_key = get_api_key()

    if not api_key:
        return "Chatbot belum dapat digunakan karena API key belum ditemukan."

    data_valid = data[[x, y]].dropna()

    if data_valid.empty:
        return "Data tidak cukup untuk menjawab pertanyaan berdasarkan hasil analisis."

    jumlah_observasi = len(data_valid)

    rata_x = data_valid[x].mean()
    rata_y = data_valid[y].mean()
    min_x = data_valid[x].min()
    max_x = data_valid[x].max()
    min_y = data_valid[y].min()
    max_y = data_valid[y].max()

    if r > 0:
        arah = "positif"
    elif r < 0:
        arah = "negatif"
    else:
        arah = "tidak menunjukkan hubungan linear"

    abs_r = abs(r)

    if abs_r < 0.20:
        kekuatan = "sangat lemah"
    elif abs_r < 0.40:
        kekuatan = "lemah"
    elif abs_r < 0.60:
        kekuatan = "sedang"
    elif abs_r < 0.80:
        kekuatan = "kuat"
    else:
        kekuatan = "sangat kuat"

    signifikansi = "signifikan" if p < 0.05 else "tidak signifikan"

    if provinsi_terpilih:
        konteks_provinsi = ", ".join(provinsi_terpilih)
    else:
        konteks_provinsi = "seluruh provinsi"

    hasil_utama = pearson_main_df.to_string(index=False)
    hasil_pendukung = pearson_support_df.to_string(index=False)

    prompt = f"""
Kamu adalah chatbot pendamping analisis pada dashboard hubungan faktor ekonomi dan kesehatan masyarakat di Indonesia.
Tugasmu adalah menjawab pertanyaan pengguna berdasarkan hasil analisis yang sedang tampil pada dashboard.

Pertanyaan pengguna:
{user_question}

Konteks filter dashboard:
- Wilayah analisis: {konteks_provinsi}
- Rentang tahun: {tahun_awal} sampai {tahun_akhir}
- Jumlah observasi: {jumlah_observasi}

Pasangan variabel yang sedang dipilih:
- Variabel X: {x}
- Variabel Y: {y}

Hasil korelasi Pearson pasangan terpilih:
- Nilai korelasi r: {r:.4f}
- Arah hubungan: {arah}
- Kekuatan hubungan: {kekuatan}
- p-value: {p:.4f}
- Signifikansi: {signifikansi}

Ringkasan statistik pasangan terpilih:
- Rata-rata {x}: {rata_x:.2f}
- Nilai minimum {x}: {min_x:.2f}
- Nilai maksimum {x}: {max_x:.2f}
- Rata-rata {y}: {rata_y:.2f}
- Nilai minimum {y}: {min_y:.2f}
- Nilai maksimum {y}: {max_y:.2f}

Tabel hasil korelasi utama:
{hasil_utama}

Tabel hasil korelasi pendukung:
{hasil_pendukung}

Batasan topik:
- Jawab hanya pertanyaan yang berkaitan dengan hasil analisis pada dashboard.
- Fokus pada korelasi Pearson, scatterplot, heatmap korelasi, nilai r, p-value, arah hubungan, kekuatan hubungan, signifikansi, tren, dan perbandingan antarprovinsi.
- Jangan membahas topik di luar data dashboard.
- Jika pertanyaan di luar konteks, jelaskan bahwa chatbot hanya menjawab berdasarkan hasil analisis dashboard.

Gaya bahasa:
- Gunakan bahasa Indonesia yang sederhana, jelas, dan mudah dipahami.
- Gunakan gaya akademik ringan, jangan terlalu teknis.
- Jika jawaban berisi beberapa temuan, gunakan poin-poin agar mudah dibaca.
- Gunakan heading singkat seperti "Ringkasan", "Hasil Utama", "Hasil Pendukung", atau "Catatan" jika diperlukan.
- Hindari paragraf panjang.
- Jawaban boleh berupa 3 sampai 6 poin singkat.
- Jangan mengulang seluruh tabel kecuali diminta pengguna.

Aturan kemudahan pemahaman:
- Jelaskan makna hasil dengan bahasa sederhana.
- Setelah menyebut nilai r atau p-value, berikan arti singkatnya.
- Jika menyebut "positif", jelaskan bahwa kedua variabel cenderung bergerak searah.
- Jika menyebut "negatif", jelaskan bahwa kedua variabel cenderung bergerak berlawanan arah.
- Jika menyebut "signifikan", jelaskan bahwa hubungan tersebut cukup kuat secara statistik pada data yang digunakan.
- Jangan hanya menyebut angka; selalu bantu pengguna memahami maknanya.

Aturan interpretasi:
- Gunakan kata “berkaitan”, “berhubungan”, “cenderung”, atau “mengindikasikan”.
- Jangan menyatakan hubungan sebab-akibat.
- Jelaskan bahwa korelasi Pearson hanya menunjukkan hubungan linear.
- Jika membahas p-value, jelaskan secara singkat apakah hasil signifikan atau tidak signifikan.
- Jika membahas Jumlah_Miskin, jelaskan bahwa variabel tersebut bersifat absolut sehingga perlu dibaca hati-hati.

Aturan kesimpulan keseluruhan:
- Jika pengguna meminta kesimpulan umum, rangkuman keseluruhan, atau seluruh hasil analisis, gunakan tabel hasil korelasi utama dan pendukung.
- Jangan hanya membahas pasangan variabel yang sedang dipilih.
- Jelaskan hasil utama dalam bentuk poin-poin.
- Untuk hasil utama, bahas P0-AHH, P0-IPM, PDRB-AHH, dan PDRB-IPM.
- Untuk hasil pendukung, bahas Jumlah_Miskin-AHH dan Jumlah_Miskin-IPM secara singkat.
- Susun jawaban dengan format:
  **Ringkasan**
  **Hasil Utama**
  **Hasil Pendukung**
  **Catatan**
- Pastikan catatan menjelaskan bahwa korelasi tidak membuktikan sebab-akibat.

Aturan interaksi:
- Jawab berdasarkan pertanyaan terbaru dan konteks hasil analisis dashboard.
- Jika pengguna meminta kesimpulan umum, rangkuman keseluruhan, atau seluruh hasil analisis, gunakan tabel hasil korelasi utama dan pendukung, bukan hanya pasangan variabel yang sedang dipilih.
- Jika pengguna hanya bertanya tentang pasangan variabel tertentu, fokuskan jawaban pada pasangan variabel yang sedang dipilih.
- Jika pengguna meminta rekomendasi lanjutan, arahkan ke eksplorasi pada halaman Tren Tahunan, Perbandingan Provinsi, scatterplot, atau heatmap korelasi.
- Jangan mengajukan lebih dari 1 pertanyaan lanjutan dalam satu jawaban.
"""

    try:
        client = genai.Client(api_key=api_key)

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )

                if response.text:
                    return response.text.strip()

                return "Chatbot belum dapat memberikan jawaban karena respons AI kosong."

            except Exception as error:
                error_text = str(error)

                if "429" in error_text or "quota" in error_text.lower() or "rate" in error_text.lower():
                    return (
                        "Chatbot belum dapat menjawab karena batas penggunaan API kemungkinan telah tercapai. "
                        "Silakan tunggu beberapa saat atau kurangi jumlah permintaan."
                    )

                if "503" in error_text or "UNAVAILABLE" in error_text:
                    if attempt < 2:
                        time.sleep(2)
                        continue

                    return (
                        "Chatbot belum dapat menjawab karena layanan AI sedang sibuk. "
                        "Silakan coba kembali beberapa saat lagi."
                    )

                if "API_KEY" in error_text or "api key" in error_text.lower() or "permission" in error_text.lower():
                    return (
                        "Chatbot belum dapat menjawab karena API key tidak valid atau belum memiliki izin akses. "
                        "Periksa kembali API key yang digunakan."
                    )

                if "model" in error_text.lower() or "not found" in error_text.lower():
                    return (
                        "Chatbot belum dapat menjawab karena model AI yang digunakan tidak tersedia. "
                        "Periksa kembali nama model pada konfigurasi."
                    )

                return f"Chatbot belum dapat menjawab. Detail error: {error_text}"

    except Exception:
        return "Chatbot belum dapat digunakan karena terjadi kendala pada proses pemanggilan AI."