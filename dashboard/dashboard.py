import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans

# ======================== Load Dataset Function ========================
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")
    numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['year_month'] = df['datetime'].dt.to_period('M').astype(str)
    return df

# ======================== Streamlit Dashboard ========================
st.title("📊 Air Quality Analysis Dashboard")
df = load_data()

# ======================== Sidebar Filter ========================
st.sidebar.header("🔧 Filter Data for All Visualizations")
st.sidebar.markdown("""
Select the date range to filter the data for all visualizations. This allows users to analyze air quality patterns within specific time periods.
""")
start_date = st.sidebar.date_input("Select Start Date:", df['datetime'].min().date())
end_date = st.sidebar.date_input("Select End Date:", df['datetime'].max().date())

if start_date > end_date:
    st.sidebar.error("Start date must be before end date!")
else:
    df = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)]

# ======================== PM2.5 Trend ========================
st.subheader("📈 Average PM2.5 Trend Across Stations")
st.markdown("""
This line chart shows the average PM2.5 concentration over time for selected stations. PM2.5 is a key air quality indicator, representing fine inhalable particles that can pose health risks.
""")
selected_stations = st.multiselect("Select Station:", df['station'].unique(), default=df['station'].unique()[:3])
filtered_df = df[df['station'].isin(selected_stations)].groupby(['year_month', 'station'])['PM2.5'].mean().reset_index()
fig_trend = px.line(filtered_df, x='year_month', y='PM2.5', color='station', markers=True)
st.plotly_chart(fig_trend)

# ======================== Air Quality per Station ========================
st.subheader("🏭 Average PM2.5 Air Quality per Station Based on Date")
st.markdown("""
This bar chart displays the average PM2.5 concentration for each station within the selected date range. It helps identify locations with higher pollution levels.
""")
kualitas_df = df.groupby('station')['PM2.5'].mean().reset_index().sort_values(by='PM2.5', ascending=False)
fig_bar_station = px.bar(kualitas_df, x='station', y='PM2.5', title="Average PM2.5 per Station", text_auto=True)
st.plotly_chart(fig_bar_station)

# ======================== Clustering Analysis Based on Pollutants ========================
st.subheader("🔎 Clustering Analysis Based on Selected Pollutants")
st.markdown("""
Select pollutants for clustering analysis. The K-Means clustering algorithm groups data points based on pollutant similarities, revealing patterns and relationships between pollution levels.
""")
pollutants = st.multiselect("Select Pollutants for Clustering:", ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], default=['PM2.5'])

if len(pollutants) > 0:
    clustering_data = df[pollutants]
    km = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = km.fit_predict(clustering_data)

    # Scatter Plot Cluster Visualization (If 2 Pollutants Selected)
    if len(pollutants) == 2:
        st.subheader("🔵 Scatter Plot Clustering Based on Pollutants")
        st.markdown("""
        This scatter plot visualizes the clustering results when two pollutants are selected. Each point represents data associated with pollutant levels, color-coded by cluster.
        """)
        fig_scatter = px.scatter(df, x=pollutants[0], y=pollutants[1], color=df['Cluster'].astype(str),
                                 title="Cluster Visualization Based on Pollutants",
                                 labels={'color': 'Cluster'})
        st.plotly_chart(fig_scatter)
    else:
        st.info("📌 Select exactly 2 pollutants to display the scatter plot.")

    # Box Plot Visualization for Each Cluster
    st.subheader("📦 Box Plot for Each Cluster")
    st.markdown("""
    These box plots show the distribution of pollutant concentrations across different clusters. This helps understand how pollutant levels vary between groups.
    """)
    for pollutant in pollutants:
        fig_box = px.box(df, x='Cluster', y=pollutant, color=df['Cluster'].astype(str),
                         title=f"{pollutant} Distribution per Cluster")
        st.plotly_chart(fig_box)

    # Data Count per Cluster Visualization
    st.subheader("📊 Data Count per Cluster")
    st.markdown("""
    This bar chart shows the number of data points assigned to each cluster, providing insight into how data is distributed among the groups.
    """)
    cluster_count = df['Cluster'].value_counts().reset_index()
    cluster_count.columns = ['Cluster', 'Count']
    fig_bar = px.bar(cluster_count, x='Cluster', y='Count', text='Count', color='Cluster',
                     title="Data Count per Cluster")
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar)
else:
    st.warning("⚠️ Select at least one pollutant for clustering analysis.")


st.sidebar.write("🚀 Built with Streamlit by T. Muhammad Caesar Maulana")
