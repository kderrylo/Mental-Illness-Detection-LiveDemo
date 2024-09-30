import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def home():
    st.markdown(
        """
# NLP Project - Mental Illness Detection with Sentiment Analysis

Repository: [Github](https://github.com/kderrylo/Mental-Illness-Detection-LiveDemo)



"""
    )

    st.markdown("""
        <style>
            .main-header {
                font-size: 32px;
                font-weight: bold;
                color: #2c3e50;
                margin-top: 20px;
                padding-bottom: 0px;
            }
            .custom-subheader {
                font-size: 20px;
                padding-top: 0;
                margin-bottom: 30px;
            }
            .section-header{
                font-size: 22px;
            }
            .content{
                font-size: 16px;
                margin-bottom:25px
            }

            .image-container{
                display: flex;
                margin: 20px 15px 20px 15px;
                flex-direction: columns;
                margin-bottom: 7px;
                width: 100%;
                height: 250px;
                justify-content: space-between;
                justify-content: center;
                align-items:center;
                gap:100px;
            }

                .image-box{
                    height: 100%;
                    width: 40%;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center
                }

                    .image{
                        height: 110%;
                        width: 110%;
                        border: 2px solid black
                    }

                    .image-caption{
                        height: 10%;
                        width: 100%;
                        display: flex;
                        justify-content: space-around;
                        align-items: center;
                        margin-top: 5px;
                        font-weight: bold;
                        font-size: 15px
                    }

                    .detail{
                        height: 60%;
                        width: 70%;
                        border: 4px solid grey
                    }

                    .detail-2{
                        height: 65%;
                        width: 125%;
                        border: 4px solid grey
                    }

                    .detail-caption{
                        height: 8%;
                        margin-right: 2%;
                        width: 125%;
                        display: flex;
                        justify-content: space-around;
                        align-items: center;
                        font-weight: bold;
                        font-size: 15px
                    }

            .section-header-2{
                margin-top: 18px;
                font-size: 22px;
            }

            .section-content-2{
                font-size: 16px;
                margin-bottom: 12px
            }

            .flowchart-box{
                display:flex;
                flex-direction: columns;
                justify-content: center;
                align-items: center;
                width: 500px;
                height: 300px;
            }

            .scaled-container{
                display: flex;
                padding: 20px 15px 20px 15px;
                flex-direction: column;
                margin-bottom: 7px;
                width: 100%;
                height: 300px;
            }

            .scaled-caption{
                height: 10%;
                width: 100%;
                display: flex;
                justify-content: space-around;
                align-items: center;
                margin-top: 5px;
                font-weight: bold;
                font-size: 13px
            }

            ul {
                list-style-type: disc;
                margin-left: 20px;
                padding-left: 0;
                font-size: 15px;
                color: #333333;
            }
            li {
                margin-bottom: 8px;
                line-height: 1.5;

            .inner-li {
                list-style-type: square;
                font-size: 15px;
            }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">Mental Illness Detection</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
            <div class="custom-subheader">
                A NLP project demo focusing on five key categories: addiction, anxiety, autism, depression, and schizophrenia. We employed a comparative approach, evaluating the effectiveness of various Machine Learing and Deep Learning models.
            </div>
            """, unsafe_allow_html=True
    )

    st.markdown('<div class="section-header">Our Objectives</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="content">
        The primary objective was to determine which approach yields the most accurate classification of mental health conditions through sentiment analysis, providing insights into the most suitable techniques for early detection.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Our Machine Learning & Deep Learning models</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li>K-Nearest Neighbours</li>
        <li>Random Forest</li>
        <li>Artificial Neural Network</li>
        <li>Convolutional Neural Network</li>
        <li>Long-Short-Term-Memory</li>
    </ul>
    """, unsafe_allow_html=True)

    df = pd.read_csv('https://raw.githubusercontent.com/kderrylo/Mental-Illness-detection-using-Sentiment-Analysis/refs/heads/master/mental_health_pre%2Bpost_pandemic_reddit.csv')
    st.markdown('<div class="section-header-2">Reddit Post Text Dataset</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Dataset obtained from Zenodo which consist of 10000 rows of post from Reddit. The dataset uses combination of pre-pandemic and post-pandemic post data.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df)

    st.markdown('<div class="section-header-2">Feature Selection</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Of all the features available in the dataset, we decided to use only 2 labels for training the model, namely subreddit as label and post as text.
    </div>
    """, unsafe_allow_html=True)

    df.drop(columns=['sent_neg', 'sent_neu',
            'sent_pos', 'sent_compound'], inplace=True)
    st.dataframe(df)

    st.markdown('<div class="section-header-2">Data Preprocessing</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Text data will be processed by removing its special characters, removing stop words, lowering words, word tokenization, lemmatizing, and features  extraction with TF-IDF. This could be benefit for the model understanding.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(pd.read_csv('Preprocessed.csv'))

    st.markdown('<div class="section-header-2">TF-IDF Vectorization</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        The text will be transformed into numeric format using TF-IDF, selecting 3000 important features, generating a matrix representing word frequencies while reducing the influence of common words. The TF-IDF matrix is combined with other features, and the data is split into training and testing sets (75% train, 25% test) for model evaluation.
    </div>
    """, unsafe_allow_html=True)

    st.image('https://imgur.com/H09xDBZ.png', caption="Vectorized Data", use_column_width=True)
    st.image('https://imgur.com/H09xDBZ.png', caption="Data Splitting for Machine Learning Models (KNN & RF)", use_column_width=True)
    


    random_forest_params = [
        ('n_estimators', 100),  
        ('max_depth', None),    
        ('min_samples_split', 2),  
        ('min_samples_leaf', 1),   
        ('random_state', 42)       
    ]

    knn_params = [
        ('n_neighbors', 5),     
        ('weights', 'uniform'),  
        ('algorithm', 'auto'),   
        ('leaf_size', 30),       
        ('p', 2)            
    ]     


    nn_params = [
        ('layers', [256, 128, 128, 64, 32]),  
        ('activation', 'relu'),             
        ('dropout', 0.5),                  
        ('kernel_regularizer', 'l2(0.01)'), 
        ('optimizer', 'adam'),              
        ('epochs', 50),                     
        ('batch_size', 32),                
        ('early_stopping_patience', 5)      
    ]


    cnn_params = [        
        ('conv_layers', [(256, 5), (128, 3), (64, 3)]),
        ('pool_size', 2),                     
        ('dropout', 0.5),                    
        ('dense_layers', [256, 128, 64]),     
        ('kernel_regularizer', 'l2(0.01)'),   
        ('optimizer', 'adam'),                
        ('epochs', 50),                       
        ('batch_size', 32),                   
        ('early_stopping_patience', 10)       
    ]

  
    lstm_params = [  
        ('lstm_layers', [128, 64, 32]),          
        ('bidirectional', True),                
        ('dropout', 0.5),                        
        ('dense_units', 32),                     
        ('activation', 'relu'),                  
        ('optimizer', 'adam'),                   
        ('epochs', 50),                          
        ('batch_size', 32),                     
        ('early_stopping_patience', 5)           
    ]

    st.markdown('<div class="section-header">Model Parameters</div>',
             unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
         we outline the key parameters utilized for each machine learning model employed in our classification tasks. Each model has distinct configurations that are critical for its performance and effectiveness in predicting mental health conditions based on the provided datasets.
    </div>
    """, unsafe_allow_html=True)

    def format_params(params):
        inner_list = []
        for param, value in params:
            inner_list.append(
                f"<li class='inner-li'><b>{param}</b> : {value}</li>")
        return "<ul>" + "".join(inner_list) + "</ul>"

    st.markdown(f"""
    <ul>
        <li><b>K-Nearest Neighbors Parameters:</b> {format_params(knn_params)}</li>
        <li><b>Random Forest Parameters:</b> {format_params(random_forest_params)}</li>
        <li><b>Artificial Neural Network (ANN) Parameters</b> {format_params(nn_params)}</li>
        <li><b>Convolutional Neural Network (CNN) Parameters:</b> {format_params(cnn_params)}</li>
        <li><b>LSTM Parameters:</b> {format_params(lstm_params)}</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header-2">Model Training and Evaluation</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        The machine learning & deep learning models are compiled with categorical cross-entropy as the loss function, and early stopping is applied during training to prevent overfitting if the validation loss stops improving.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://imgur.com/Ntss7qb.png" alt="KNN Matrix">
                <div class="image-caption">KNN Matrix</div>
        </div>
        <div class="image-box">
                <img class="detail-2" src="https://imgur.com/6yN5PGl.png" alt="KNN Evaluation"> 
                <div class="detail-caption">KNN Evaluation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://imgur.com/pZNRSEd.png" alt="Random Forest Matrix">
                <div class="image-caption">Random Forest Matrix</div>
        </div>
        <div class="image-box">
                <img class="detail-2" src="https://imgur.com/laejP75.png" alt="Random Forest Evaluation"> 
                <div class="detail-caption">Random Forest Evaluation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://imgur.com/CcjfA8N.png" alt="">
                <div class="image-caption">ANN Accuracy Plot</div>
        </div>
        <div class="image-box">
                <img class="image" src="https://imgur.com/4sAceXO.png" alt="">
                <div class="image-caption">ANN MSE Plot</div>
        </div>
    </div>
    <div class="image-container">
        <div class="image-box">
                <img class="detail-2" src="https://imgur.com/ihKqMKg.png" alt="">
                <div class="image-caption">ANN Evaluation</div>
        </div>
        <div class="image-box">
                <img class="detail-2" src="https://imgur.com/G4MsXZO.png" alt="">
                <div class="image-caption">ANN Classification Report</div>
        </div>
    </div>
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://imgur.com/BP3bw9C.png" alt="">
                <div class="image-caption">ANN Matrix</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://imgur.com/848pdiM.png" alt="">
                <div class="image-caption">CNN Accuracy Plot</div>
        </div>
        <div class="image-box">
                <img class="image" src="https://imgur.com/X6VI95x.png" alt="">
                <div class="image-caption">CNN MSE Plot</div>
        </div>
    </div>
    <div class="image-container">
        <div class="image-box">
                <img class="detail-2" src="https://imgur.com/VvzBepe.png" alt="">
                <div class="image-caption">CNN Evaluation</div>
        </div>
        <div class="image-box">
                <img class="detail-2" src="https://imgur.com/Yfe8GbR.png" alt="">
                <div class="image-caption">CNN Classification Report</div>
        </div>
    </div>
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://imgur.com/tAqiIST.png" alt="">
                <div class="image-caption">CNN Matrix</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://imgur.com/wXQiJt8.png" alt="">
                <div class="image-caption">LSTM Accuracy Plot</div>
        </div>
        <div class="image-box">
                <img class="image" src="https://imgur.com/4IzXjXs.png" alt="">
                <div class="image-caption">LSTM MSE Plot</div>
        </div>
    </div>
    <div class="image-container">
        <div class="image-box">
                <img class="detail-2" src="https://imgur.com/ZtdVWVe.png" alt="">
                <div class="image-caption">LSTM Evaluation</div>
        </div>
        <div class="image-box">
                <img class="detail-2" src="https://imgur.com/raJbW7l.png" alt="">
                <div class="image-caption">LSTM Classification Report</div>
        </div>
    </div>
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://imgur.com/CXs2q5W.png" alt="">
                <div class="image-caption">LSTM Matrix</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header-2">Flowchart</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Following is a flowchart that manifests the steps to create a machine learning & deep learning models.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="image-container">
        <img class="flowchart-box" src="https://imgur.com/HSDBFK2.png" alt="Flowchart"> 
    </div>
    """, unsafe_allow_html=True)

    return
