import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")  # Sesuaikan dengan path dataset

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

# ======================== Analisis RFM Berdasarkan Data Kualitas Udara ========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ======================== Fungsi Memuat Dataset ========================
def load_data():
    df = pd.read_csv("main_data.csv")  # Sesuaikan dengan path dataset
    numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['year_month'] = df['datetime'].dt.to_period('M').astype(str)
    return df

# ======================== Analisis RFM untuk Kualitas Udara ========================
st.subheader("📊 Analisis RFM (Recency, Frequency, Monetary) untuk Kualitas Udara")
df = load_data()
st.write("Data Awal:", df.head())

# ======================== Menghitung Recency ========================
df_recency = df.groupby(by='station', as_index=False)['datetime'].max()
df_recency.columns = ['Station', 'LastObservationDate']
recent_date = df_recency['LastObservationDate'].max()
df_recency['Recency'] = df_recency['LastObservationDate'].apply(lambda x: (recent_date - x).days)

# ======================== Menghitung Frequency ========================
frequency_df = df.groupby(by='station', as_index=False)['datetime'].nunique()
frequency_df.columns = ['Station', 'Frequency']

# ======================== Menghitung Monetary ========================
monetary_df = df.groupby(by='station', as_index=False)['PM2.5'].mean()
monetary_df.columns = ['Station', 'Mean_PM2.5']

# ======================== Menggabungkan Ketiga Kolom ========================
rf_df = df_recency.merge(frequency_df, on='Station')
rfm_df = rf_df.merge(monetary_df, on='Station').drop(columns='LastObservationDate')
st.write("Data RFM:", rfm_df.head())

# ======================== Peringkat dan Normalisasi ========================
rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending=True)
rfm_df['M_rank'] = rfm_df['Mean_PM2.5'].rank(ascending=True)

rfm_df['R_rank_norm'] = (rfm_df['R_rank'] / rfm_df['R_rank'].max()) * 100
rfm_df['F_rank_norm'] = (rfm_df['F_rank'] / rfm_df['F_rank'].max()) * 100
rfm_df['M_rank_norm'] = (rfm_df['M_rank'] / rfm_df['M_rank'].max()) * 100

rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

# ======================== Menghitung Skor RFM ========================
rfm_df['RFM_Score'] = 0.15 * rfm_df['R_rank_norm'] + 0.28 * rfm_df['F_rank_norm'] + 0.57 * rfm_df['M_rank_norm']
rfm_df['RFM_Score'] *= 0.05
rfm_df = rfm_df.round(2)

# ======================== Segmentasi Berdasarkan Skor RFM ========================
rfm_df["Air_Quality_Segment"] = np.where(rfm_df['RFM_Score'] > 4.5, "Excellent",
                                  np.where(rfm_df['RFM_Score'] > 4, "Good",
                                  np.where(rfm_df['RFM_Score'] > 3, "Moderate",
                                  np.where(rfm_df['RFM_Score'] > 1.6, "Poor", "Very Poor"))))

st.write("Segmen Kualitas Udara Berdasarkan RFM:", rfm_df[['Station', 'RFM_Score', 'Air_Quality_Segment']].head(20))

# ======================== Visualisasi Segmen Kualitas Udara ========================
fig, ax = plt.subplots()
plt.pie(rfm_df.Air_Quality_Segment.value_counts(),
        labels=rfm_df.Air_Quality_Segment.value_counts().index,
        autopct='%.0f%%', startangle=140)
plt.title("Distribusi Segmen Kualitas Udara Berdasarkan Analisis RFM")
st.pyplot(fig)

st.markdown("""
**Penjelasan Analisis RFM untuk Kualitas Udara:**
- **Recency (Keterkinian):** Mengukur berapa lama sejak pengamatan terakhir dilakukan di stasiun.
- **Frequency (Frekuensi):** Mengukur seberapa sering stasiun mencatat data kualitas udara.
- **Monetary (Nilai Moneter - Diadaptasi sebagai Rata-rata PM2.5):** Mengukur tingkat polusi rata-rata PM2.5 di setiap stasiun.

**Kategori Kualitas Udara:**
- **Excellent:** Skor RFM > 4.5
- **Good:** 4.5 > Skor RFM > 4
- **Moderate:** 4 > Skor RFM > 3
- **Poor:** 3 > Skor RFM > 1.6
- **Very Poor:** Skor RFM < 1.6
""")


# ======================== Tampilkan Data ========================
if st.checkbox("📜 Tampilkan Data Awal"):
    st.write(df.head())

st.sidebar.write("🚀 Dibuat dengan Streamlit oleh T. Muhammad Caesar Maulana")
