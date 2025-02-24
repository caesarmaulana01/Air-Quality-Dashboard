import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    df = pd.read_csv("/mount/src/air-quality-dashboard/dashboard/main_data.csv")  # Sesuaikan dengan path dataset

    # Konversi kolom numerik
    numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # Buat kolom datetime
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['year_month'] = df['datetime'].dt.to_period('M').astype(str)

    return df

df = load_data()

st.title("📊 Dashboard Analisis Kualitas Udara")

# ======================== Tren PM2.5 ========================

st.subheader("📈 Tren Rata-rata PM2.5 di Berbagai Stasiun")
selected_stations = st.multiselect("Pilih Stasiun:", df['station'].unique(), default=df['station'].unique()[:3])
filtered_df = df[df['station'].isin(selected_stations)].groupby(['year_month', 'station'])['PM2.5'].mean().reset_index()
fig_trend = px.line(filtered_df, x='year_month', y='PM2.5', color='station', markers=True)
st.plotly_chart(fig_trend)

# ======================== Korelasi Faktor ========================
st.subheader("📊 Korelasi Faktor terhadap PM2.5")
correlation_matrix = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].corr()
fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
st.pyplot(fig_corr)


## ======================== Analisis Clustering Sederhana Berdasarkan Tingkat PM2.5 (Pertahun) ========================

st.subheader("🔎 Analisis Clustering Sederhana Berdasarkan Tingkat PM2.5 (Pertahun)")

# Dropdown untuk memilih tahun
selected_year = st.selectbox("Pilih Tahun:", sorted(df['year'].unique()))

# Agregasi statistik per daerah untuk tahun yang dipilih (Mean dan Median)
df_year = df[df['year'] == selected_year].groupby('station').agg({
    'PM2.5': ['mean', 'median']
}).reset_index()
df_year.columns = ['Station', 'Mean_PM2.5', 'Median_PM2.5']

# Visualisasi kualitas udara per daerah
fig, ax = plt.subplots(figsize=(12, 6))
df_melted = df_year.melt(id_vars='Station', var_name='Statistic', value_name='PM2.5')
sns.barplot(data=df_melted, x='Station', y='PM2.5', hue='Statistic')
plt.title(f"Kualitas Udara (PM2.5) di Setiap Daerah pada Tahun {selected_year} (Rata-rata dan Median)")
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("""
**Penjelasan Analisis:**
- **Mean (Rata-rata):** Menggambarkan nilai rata-rata PM2.5 di setiap stasiun.
- **Median:** Memberikan indikasi distribusi data yang robust terhadap outlier.
- **Menggunakan Teknik Rata-rata dan Median:** Pemilihan mean dan median memberikan pemahaman yang seimbang terhadap distribusi data dan potensi outlier.
""")

# ======================== Analisis Kualitas Udara Pertahunnya Berdasarkan Stasiun ========================

st.subheader("📆 Analisis Kualitas Udara Pertahunnya Berdasarkan Stasiun")

# Dropdown untuk memilih stasiun
selected_station = st.selectbox("Pilih Stasiun:", sorted(df['station'].unique()))

# Agregasi statistik pertahun untuk stasiun yang dipilih (Mean dan Median)
df_station = df[df['station'] == selected_station].groupby('year').agg({
    'PM2.5': ['mean', 'median']
}).reset_index()
df_station.columns = ['Year', 'Mean_PM2.5', 'Median_PM2.5']

# Visualisasi kualitas udara pertahunnya
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_station, x='Year', y='Mean_PM2.5', marker='o', label='Mean')
sns.lineplot(data=df_station, x='Year', y='Median_PM2.5', marker='o', label='Median')

plt.title(f"Tren Kualitas Udara (PM2.5) di Stasiun {selected_station} Pertahunnya (Rata-rata dan Median)")
plt.ylabel("Konsentrasi PM2.5 (µg/m³)")
plt.xlabel("Tahun")
plt.legend()
st.pyplot(fig)

st.markdown(f"""
**Penjelasan Analisis:**
- Grafik ini menunjukkan bagaimana kualitas udara di stasiun **{selected_station}** berubah dari tahun ke tahun.
- **Mean (Rata-rata)** dan **Median** digunakan untuk memberikan pemahaman yang seimbang terhadap data dan kemungkinan adanya outlier.
- Analisis ini menggunakan teknik statistik sederhana (mean dan median) untuk memberikan gambaran umum pola polusi udara.
""")

# ======================== Dashboard Streamlit ========================
st.subheader("🌫️ Analisis Kualitas Udara Berdasarkan AQI dengan Clustering dan Binning")
df = load_data()

# ======================== Filter Tanggal dan Parameter ========================
try:
    start_date = st.date_input("Pilih Tanggal Mulai:", df['datetime'].min().date())
    end_date = st.date_input("Pilih Tanggal Akhir:", df['datetime'].max().date())
    
    if start_date > end_date:
        st.error("Tanggal mulai harus sebelum tanggal akhir!")
    else:
        df = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)]
except:
    st.warning("Tanggal tidak valid. Menampilkan semua data.")

pollutants = st.multiselect("Pilih Jenis Polutan:", ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], default=['PM2.5'])
temp_range = st.slider("Rentang Temperatur (°C):", float(df['TEMP'].min()), float(df['TEMP'].max()), (float(df['TEMP'].min()), float(df['TEMP'].max())))
wind_range = st.slider("Rentang Kecepatan Angin (m/s):", float(df['WSPM'].min()), float(df['WSPM'].max()), (float(df['WSPM'].min()), float(df['WSPM'].max())))

df = df[(df['TEMP'] >= temp_range[0]) & (df['TEMP'] <= temp_range[1]) &
        (df['WSPM'] >= wind_range[0]) & (df['WSPM'] <= wind_range[1])]

# ======================== Manual Grouping Berdasarkan AQI ========================
def classify_aqi(pm25):
    if pm25 <= 50:
        return "Excellent"
    elif pm25 <= 100:
        return "Good"
    elif pm25 <= 150:
        return "Lightly Polluted"
    elif pm25 <= 200:
        return "Moderately Polluted"
    elif pm25 <= 300:
        return "Heavily Polluted"
    else:
        return "Severely Polluted"

df['Air_Quality_Category'] = df['PM2.5'].apply(classify_aqi)

# ======================== Clustering Berdasarkan Parameter ========================
clustering_data = df[pollutants + ['TEMP', 'WSPM']]
km = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = km.fit_predict(clustering_data)

# ======================== Visualisasi Distribusi Kualitas Udara ========================
color_palette = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600']
fig, ax = plt.subplots()
plt.pie(df['Air_Quality_Category'].value_counts(),
        labels=df['Air_Quality_Category'].value_counts().index,
        autopct='%.0f%%', startangle=140,
        colors=color_palette)
plt.title("Distribusi Kategori Kualitas Udara")
st.pyplot(fig)

st.write("**Detail Kategori Kualitas Udara:**")
st.dataframe(df[['datetime', 'station'] + pollutants + ['TEMP', 'WSPM', 'Air_Quality_Category', 'Cluster']].head())

st.markdown("""
**Kategori Kualitas Udara (Berdasarkan PM2.5):**
- **Excellent (Sangat Baik):** PM2.5 ≤ 50
- **Good (Baik):** 51 - 100
- **Lightly Polluted (Tercemar Ringan):** 101 - 150
- **Moderately Polluted (Tercemar Sedang):** 151 - 200
- **Heavily Polluted (Tercemar Berat):** 201 - 300
- **Severely Polluted (Sangat Tercemar):** > 300

**Fitur Tambahan:**
- Filter berdasarkan tanggal, polutan, temperatur, dan kecepatan angin.
- Clustering berdasarkan parameter polutan, temperatur, dan angin.
- Visualisasi kategori kualitas udara dengan kombinasi warna tertentu.
""")

# ======================== Tampilkan Data ========================
if st.checkbox("📜 Tampilkan Data Awal"):
    st.write(df.head())

st.sidebar.write("🚀 Dibuat dengan Streamlit oleh T. Muhammad Caesar Maulana")
