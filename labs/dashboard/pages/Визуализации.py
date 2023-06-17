import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import seaborn as sea
import pandas as pd
import random
import numpy as np

def visualization_page() -> None:
    st.title("Зависимости между признаками")
    st.header("Тепловая карта корреляции признаков.")
    heatmap = get_heatmap()
    st.pyplot(heatmap)

    st.header("Корреляция между ценой бриллианта и его признаками.")
    hist = get_correlations_histogram()
    st.pyplot(hist)

    st.header("Попарные зависимости цены от значений признаков.")
    paiwise = get_pairwise()
    st.pyplot(paiwise)
    return

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("./labs/dashboard/regression_preprocessed.csv")

@st.cache_data
def get_heatmap() -> fig.Figure:
    data: pd.DataFrame = load_data()
    correlations = data.corr()
    figure = plt.figure()
    sea.heatmap(correlations, cmap="plasma")
    return figure

@st.cache_data
def get_correlations_histogram() -> fig.Figure:
    data: pd.DataFrame = load_data()
    correlations = data.drop(columns=["price"]).corrwith(data["price"])
    figure = plt.figure(figsize=(18, 10))
    columns = data.drop(columns=["price"]).columns.map(lambda x: x[8:] if x.startswith("clarity_") else x)
    sea.pointplot(x=columns, y=correlations)
    plt.axhline(y=0, color='gray', linestyle='--')
    return figure

@st.cache_data
def get_pairwise() -> fig.Figure:
    data: pd.DataFrame = load_data()
    figure, _ = plt.subplots(3, 2, figsize=(10, 15))
    axes = figure.axes
    features = data.drop(columns="price").columns[:7].to_list()
    features.pop(1)
    for i, feature in enumerate(features):
        axes[i].scatter(data[feature], data.price, color=[random.randint(20, 200) / 255 for _ in range(3)])
    return figure

st.set_page_config(page_title="Визуализации")

visualization_page()
