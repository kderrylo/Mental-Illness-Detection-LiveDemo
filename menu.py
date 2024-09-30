import streamlit as st
from homepage import home
from eda_page import eda
from preprocessing_page import preprocessing
from training_page import training
from training_ann import training_ann
from streamlit_option_menu import option_menu

def router(key,page="Home"):
    match page or st.session_state[key]:
        case "Home":
            home()
        case "EDA":
            eda()
        case "Training & Predict K-NN/RF":
            training()
        case "Training & Predict ANN":
            training_ann()
    

def show_menu():
    with st.sidebar:
        menu = option_menu("Mental Illness Detection", ["Home","EDA", "Training & Predict K-NN/RF", "Training & Predict ANN"],
            icons=['house', 'list-task'], key="menu", default_index=0)
    router("menu",menu)    