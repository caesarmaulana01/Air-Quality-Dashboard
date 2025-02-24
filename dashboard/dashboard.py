import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans

# ======================== Fungsi Memuat Dataset ========================
@st.cache_data
def load_data():
    df = pd.read_csv("/mount/src/air-quality-dashboard/dashboard/dashboard.py")
    numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['year_month'] = df['datetime'].dt.to_period('M').astype(str)
    return df

# ======================== Dashboard Streamlit ========================
st.title("📊 Dashboard Analisis Kualitas Udara")
df = load_data()

# ======================== Sidebar Filter ========================
st.sidebar.header("🔧 Filter Data untuk Semua Visualisasi")
start_date = st.sidebar.date_input("Pilih Tanggal Mulai:", df['datetime'].min().date())
end_date = st.sidebar.date_input("Pilih Tanggal Akhir:", df['datetime'].max().date())

if start_date > end_date:
    st.sidebar.error("Tanggal mulai harus sebelum tanggal akhir!")
else:
    df = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)]

# ======================== Tren PM2.5 ========================
st.subheader("📈 Tren Rata-rata PM2.5 di Berbagai Stasiun")
selected_stations = st.multiselect("Pilih Stasiun:", df['station'].unique(), default=df['station'].unique()[:3])
filtered_df = df[df['station'].isin(selected_stations)].groupby(['year_month', 'station'])['PM2.5'].mean().reset_index()
fig_trend = px.line(filtered_df, x='year_month', y='PM2.5', color='station', markers=True)
st.plotly_chart(fig_trend)

# ======================== Kualitas Udara per Stasiun ========================
st.subheader("🏭 Kualitas Udara Rata-rata PM2.5 per Stasiun Berdasarkan Tanggal")
kualitas_df = df.groupby('station')['PM2.5'].mean().reset_index().sort_values(by='PM2.5', ascending=False)
fig_bar_station = px.bar(kualitas_df, x='station', y='PM2.5', title="Rata-rata PM2.5 per Stasiun", text_auto=True)
st.plotly_chart(fig_bar_station)

# ======================== Analisis Clustering Berdasarkan Polutan ========================
st.subheader("🔎 Analisis Clustering Berdasarkan Polutan yang Dipilih")
pollutants = st.multiselect("Pilih Jenis Polutan untuk Clustering:", ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], default=['PM2.5'])

if len(pollutants) > 0:
    clustering_data = df[pollutants]
    km = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = km.fit_predict(clustering_data)

    # Visualisasi Scatter Plot Cluster (Jika Ada 2 Polutan Terpilih)
    if len(pollutants) == 2:
        st.subheader("🔵 Scatter Plot Clustering Berdasarkan Polutan")
        fig_scatter = px.scatter(df, x=pollutants[0], y=pollutants[1], color=df['Cluster'].astype(str),
                                 title="Visualisasi Cluster Berdasarkan Polutan",
                                 labels={'color': 'Cluster'})
        st.plotly_chart(fig_scatter)
    else:
        st.info("📌 Pilih tepat 2 polutan untuk menampilkan scatter plot.")

    # Visualisasi Box Plot untuk Setiap Cluster
    st.subheader("📦 Box Plot untuk Setiap Cluster")
    for pollutant in pollutants:
        fig_box = px.box(df, x='Cluster', y=pollutant, color=df['Cluster'].astype(str),
                         title=f"Distribusi {pollutant} per Cluster")
        st.plotly_chart(fig_box)

    # Visualisasi Jumlah Data per Cluster
    st.subheader("📊 Jumlah Data per Cluster")
    cluster_count = df['Cluster'].value_counts().reset_index()
    cluster_count.columns = ['Cluster', 'Count']
    fig_bar = px.bar(cluster_count, x='Cluster', y='Count', text='Count', color='Cluster',
                     title="Jumlah Data per Cluster")
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar)
else:
    st.warning("⚠️ Pilih setidaknya satu polutan untuk analisis clustering.")

# ======================== Tampilkan Data ========================
if st.checkbox("📜 Tampilkan Data Awal"):
    display_cols = ['datetime', 'station'] + pollutants + ['Cluster'] if 'Cluster' in df.columns else []
    st.write(df[display_cols].head())

st.sidebar.write("🚀 Dibuat dengan Streamlit oleh T. Muhammad Caesar Maulana")
