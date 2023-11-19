import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import matplotlib.pyplot as plt  # for bar plot

dataset_url = "https://caiomc03bucket.blob.core.windows.net/cont/datacsv.csv"

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url, sep=";")

df = get_data()

# dashboard title
st.title("Real-Time / Live Data Science Dashboard")

job_filter = st.selectbox("Select the Job", df)

# creating a single-element container
placeholder = st.empty()

# near real-time / live feed simulation

fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.markdown("### First Chart")
    fig = px.density_heatmap(data_frame=df, y="id", x="valor")
    st.write(fig)

import matplotlib.pyplot as plt

with fig_col2:
    st.markdown("### Second Chart")
    fig, ax = plt.subplots()
    ax.bar(df["id"],df["valor"])

    st.pyplot(fig)
        

st.markdown("### Detailed Data View")
st.dataframe(df)
time.sleep(1)