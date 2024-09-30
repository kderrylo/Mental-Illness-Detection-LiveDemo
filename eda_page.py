
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def eda():
    st.title("Exploratory Data Analysis")
    df=pd.read_csv('https://raw.githubusercontent.com/kderrylo/Mental-Illness-detection-using-Sentiment-Analysis/refs/heads/master/mental_health_pre%2Bpost_pandemic_reddit.csv')
    st.write("Dataframe:")
    st.write(df)
    st.write("Data type in each columns:")
    st.write(df.dtypes)
    
    st.write("Data distribution for each label")
    st.image('https://imgur.com/ZEnTVI4.png', caption="The data is well-balanced", use_column_width=True)

    st.write("The length of the texts posts are (in terms of word count) for each label and overall.")
    st.image('https://imgur.com/XcBbMBP.png', use_column_width=True)

    st.write("The most common words across the dataset.")
    st.image('https://imgur.com/YlDj3s2.png', use_column_width=True)

    st.write("Wordcloud visualization for all texts in the dataset")
    st.image('https://imgur.com/rZRyL0A.png', use_column_width=True)

    st.markdown('---')

    st.write("Wordcloud visualization for each category/class/label")
    st.image('https://imgur.com/iByuDsB.png', use_column_width=True)
    st.image('https://imgur.com/Sh7KrON.png', use_column_width=True)
    st.image('https://imgur.com/R1eHSQ9.png', use_column_width=True)
    st.image('https://imgur.com/VpBTedp.png', use_column_width=True)
    st.image('https://imgur.com/P5by1n7.png', use_column_width=True)

    
        
    return
