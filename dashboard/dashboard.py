"""
Dashboard Analisis Kualitas Udara

Dibuat oleh: T. Muhammad Caesar Maulana

Deskripsi:
Dashboard ini menggunakan Streamlit untuk menganalisis tren dan distribusi data kualitas udara berdasarkan dataset yang tersedia.
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")  # Sesuaikan dengan path dataset

    # Konversi kolom numerik
    numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Hapus baris yang mengandung NaN
    df = df.dropna()

    # Buat kolom datetime
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

    # Ubah format waktu menjadi string agar seaborn bisa memproses
    df['year_month'] = df['datetime'].dt.to_period('M').astype(str)

    # Agregasi rata-rata bulanan PM2.5 di setiap stasiun
    df_monthly = df.groupby(['year_month', 'station'])['PM2.5'].mean().reset_index()

    # Konversi kembali agar memastikan 'PM2.5' bertipe numerik
    df_monthly['PM2.5'] = pd.to_numeric(df_monthly['PM2.5'], errors='coerce')

    # Pastikan tidak ada NaN
    df_monthly = df_monthly.dropna()

    return df, df_monthly

df, df_monthly = load_data()

# Streamlit App
st.title("📊 Dashboard Analisis Kualitas Udara")

# Sidebar: Pilihan Analisis
st.sidebar.header("🔍 Pilihan Analisis")
analysis_type = st.sidebar.radio("Pilih Analisis:", ["Tren PM2.5", "Distribusi Data"])

if analysis_type == "Tren PM2.5":
    st.subheader("📈 Tren PM2.5 di Berbagai Stasiun")
    
    # Pilih stasiun untuk ditampilkan
    selected_stations = st.multiselect("Pilih Stasiun Pemantauan:", df_monthly['station'].unique(), default=df_monthly['station'].unique()[:3])
    
    # Filter data berdasarkan pilihan pengguna
    filtered_df = df_monthly[df_monthly['station'].isin(selected_stations)]

    # Plot Tren Waktu PM2.5
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=filtered_df, x='year_month', y='PM2.5', hue='station', marker='o', palette="tab10")
    plt.xticks(rotation=45)
    plt.xlabel("Waktu (Bulan)")
    plt.ylabel("Konsentrasi PM2.5 (µg/m³)")
    plt.title("Tren PM2.5 di Berbagai Daerah")
    plt.legend(title="Stasiun Pemantauan", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

elif analysis_type == "Distribusi Data":
    st.subheader("📊 Analisis Distribusi Data")
    col = st.selectbox("Pilih variabel:", ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'])

    # Plot Distribusi
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col], bins=30, kde=True, ax=ax)
    plt.xlabel(col)
    plt.ylabel("Frekuensi")
    plt.title(f"Distribusi {col}")
    st.pyplot(fig)

# Menampilkan DataFrame
if st.checkbox("📜 Tampilkan Data"):
    st.write(df.head())

st.sidebar.write("🚀 Dibuat dengan Streamlit oleh T. Muhammad Caesar Maulana")
