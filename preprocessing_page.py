import streamlit as st
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
import pandas as pd
from streamlit import session_state as state

scaler_options = {"Standard Scaler": StandardScaler(), "Robust Scaler": RobustScaler(), "MinMax Scaler": MinMaxScaler()}
binary_map = {'no': 0, 'yes': 1}

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def preprocessing():
    if 'preprocessing_done' not in state:
        state.preprocessing_done = False
    st.title("Preprocessing")
    
    df = pd.read_csv('student-mat.csv', sep=';', usecols=['sex', 'age', 'address', 'Medu', 'Fedu', 
     'traveltime', 'failures', 'paid', 'higher', 'internet', 'goout', 'G1', 'G2', 'G3'])

    st.write("Select the scaler you want to use:")
    scaler = st.selectbox("Select Scaler", list(scaler_options.keys()))
    remove_outliers_option = st.checkbox("Remove Outliers", False)

    if st.button("Preprocess"):
        state.preprocessing_done = False
        state.scaler_option = scaler
        state.encoder = {}
        
        for column in ['paid', 'higher', 'internet']:
            df[column] = df[column].map(binary_map)

        for column in ['address', 'sex', 'paid', 'higher', 'internet']:
            state.encoder[column] = LabelEncoder().fit(df[column])
            df[column] = state.encoder[column].transform(df[column])

        num_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'failures', 'goout', 'G1', 'G2']
        
        if remove_outliers_option:
            for column in num_columns:
                df = remove_outliers(df, column)
        
        result = df[num_columns]
        state.scaler = scaler_options[scaler].fit(result)
        scaled_df = state.scaler.transform(result)

        df = df.copy()
        df[result.columns] = scaled_df
        state.preprocessing_done = True
        state.df = df
        st.success("Preprocessing Done")
        st.dataframe(df)
    return
