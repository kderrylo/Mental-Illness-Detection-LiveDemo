import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from menu import show_menu
import pickle
import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

        
def main():
        # random_button_callback(True)
    # main_render()
    show_menu()
    return 0

if __name__ == '__main__':
    main()
