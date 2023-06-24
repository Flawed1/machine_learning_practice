import streamlit as st
from typing import *
import pickle
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import tensorflow as tf
import xgboost as xgb
import sklearn.preprocessing as preprocessing


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("./labs/dashboard/regression_preprocessed.csv")

@st.cache_data
def get_cut_options(data: pd.DataFrame) -> Iterable[str]:
    options = data.cut.unique()
    options.sort()
    return options

@st.cache_data
def get_clarity_options(data: pd.DataFrame) -> Iterable[str]:
    return [column[8:] for column in data.columns.to_numpy() if column.startswith("clarity_")]

@st.cache_data
def get_color_options(data: pd.DataFrame) -> Iterable[str]:
    return [column[6:] for column in data.columns.to_numpy() if column.startswith("color_")]

def restore_data(data: NDArray, input_data: NDArray) -> NDArray:
    restored_data = list()
    restored_data.extend(input_data[:7])
    for option in get_clarity_options(data):
        restored_data.append(float(input_data[7] == option))
    for option in get_color_options(data):
        restored_data.append(float(input_data[8] == option))
    return np.array(restored_data, dtype=float).reshape(1, -1)

def predict(model: Any, data: NDArray, input_data: NDArray, preprocessor: Optional[Any]) -> float:
    input_data = restore_data(data, input_data) 
    if preprocessor:
        input_data = preprocessor.fit_transform(input_data)
    return model.predict(input_data)

@st.cache_data
def assign_default_model() -> str:
    selected_model: str = "Линейная модель"

def model_page() -> None:
    data = load_data()
    number_input_defaults = {"min_value": 0., "step": 0.01}

    st.title("Модели")
    selected_model = st.selectbox("Модель", available_models.keys())
    preprocessor: Optional[Any] = available_models.get(selected_model).get("Preprocessor")
    st.text(available_models.get(selected_model).get("Description"))
    carat = st.number_input(label="Carat", **number_input_defaults)
    cut = st.selectbox("Cut", get_cut_options(data))
    depth = st.number_input(label="Depth", **number_input_defaults)
    table = st.number_input(label="Table", **number_input_defaults)
    x_col, y_col, z_col = st.columns(3)
    x = x_col.number_input(label="x", **number_input_defaults)
    y = y_col.number_input(label="y", **number_input_defaults)
    z = z_col.number_input(label="z", **number_input_defaults)
    color = st.selectbox("Color", get_color_options(data))
    clarity = st.selectbox("Clarity", get_clarity_options(data))
    if st.button("Предсказать"):
        prediction = predict(available_models.get(selected_model).get("Model"),
                             data,
                             np.array([carat, cut, depth, table, x, y, z, color, clarity]),
                             preprocessor)
        st.markdown("__Price:__ {:.2f}".format(float(prediction)))

def load_object(filepath: str) -> Any:
    if filepath.endswith("pickle"):
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
    elif filepath.endswith("json"):
        model = xgb.XGBRegressor()
        model.load_model(filepath)
    else:
        model = tf.keras.saving.load_model(filepath)
    return model

@st.cache_data
def get_available_models() -> dict[str, dict[str, Any]]:
    available_models = {
        "Линейная модель": {
            "Description": "Модель полиномиальной регрессии с регуляризацией Ridge.",
            "Model": load_object("./labs/dashboard/polymodel.pickle"),
            "Preprocessor": preprocessing.PolynomialFeatures(3)
        },
        "Нейронная сеть": {
            "Description": "Модель, основанная на нейронной сети с помощью Tensorflow.",
            "Model": load_object("./labs/dashboard/tfmodel.keras")
        },
        "XGBoost модель": {
            "Description": "Модель, использующая градиентный бустинг из библиотеки XGBoost.",
            "Model": load_object("./labs/dashboard/xgbmodel.json")
        }
    }
    return available_models

available_models: dict[str, dict[str, Any]] = get_available_models()
selected_model: str = "Линейная модель"
model_page()
