import streamlit as st
from streamlit import session_state as state
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

model_options = {
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Random Forest": RandomForestClassifier,
}

param_options = {
    'n_estimators': [1, 1000, 100],
    'max_depth': [1, 100, 3],
    'min_samples_split': [2, 100, 2],
    'min_samples_leaf': [1, 100, 1],
    'n_neighbors': [1, 15, 5],
}

def pre_processing(text_data):
    return text_data  

def train():
    params = {}

    model_option = st.selectbox("Model:", list(model_options.keys()))
    params['test_size'] = st.slider("Test Dataset Size", 0.05, 0.9, 0.2)
    params['random_state'] = st.number_input("Random State", 0, 100, 42)

    params = dict(params, **dict([(k, st.number_input(k, min_value=r[0], max_value=r[1], value=r[2])) for k, r in param_options.items()
                                  if k in model_options[model_option].__init__.__code__.co_varnames]))

    text_list = pd.read_csv('Preprocessed.csv')
    text_list.columns = ['prepro']
    data = pd.read_csv(
        'https://raw.githubusercontent.com/kderrylo/Mental-Illness-detection-using-Sentiment-Analysis/refs/heads/master/mental_health_pre%2Bpost_pandemic_reddit.csv')
    
    state.vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    sparse_matrix = state.vectorizer.fit_transform(text_list['prepro']).toarray()

    state.data = data

    X = np.hstack((sparse_matrix, data.drop(columns=['post', 'subreddit']).to_numpy()))
    y = data["subreddit"].values

    state.label_encoder = LabelEncoder()
    state.y_train = state.label_encoder.fit_transform(y)

    if st.button("Train"):
        state.training_done = False  
        model_params = model_options[model_option].__init__.__code__.co_varnames

        state.x_train, state.x_test, state.y_train, state.y_test = train_test_split(
            X, state.y_train, test_size=params['test_size'], random_state=params['random_state'])

        model_params = {k: v for k, v in params.items() if k in model_params}

        with st.spinner('Training the model...'):
            state.model = model_options[model_option](**model_params).fit(state.x_train, state.y_train)

        y_pred = state.model.predict(state.x_test)

        accuracy = accuracy_score(state.y_test, y_pred)
        precision = precision_score(state.y_test, y_pred, average='weighted')
        recall = recall_score(state.y_test, y_pred, average='weighted')
        f1 = f1_score(state.y_test, y_pred, average='weighted')

        mse = mean_squared_error(state.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(state.y_test, y_pred)
        r2 = r2_score(state.y_test, y_pred)

        report = classification_report(state.y_test, y_pred, target_names=state.label_encoder.classes_)

        st.write("### Model Evaluation Metrics")
        st.markdown("---")
        st.write("Accuracy: ", accuracy)
        st.write("Precision: ", precision)
        st.write("Recall: ", recall)
        st.write("F1-Score: ", f1)
        st.markdown("---")
        st.write("Mean Squared Error (MSE): ", mse)
        st.write("Root Mean Squared Error (RMSE): ", rmse)
        st.write("Mean Absolute Error (MAE): ", mae)
        st.write("R² Score: ", r2)
        st.markdown("---")
        st.write("Classification Report:\n")
        st.text(report)
        st.markdown("---")

        st.success("Training Done")
        state.training_done = True  

def test_real_data():
    st.markdown("Examples:")
    st.markdown("- I often feel overwhelmed by feelings of worry and fear. It’s hard to focus on anything without feeling anxious.")
    st.markdown("- I’ve been feeling really low lately, and I struggle to find joy in activities I used to love. I feel tired all the time.")
    st.markdown("- I find it hard to resist the urge to check my phone constantly. I often feel like I need it to function properly.")
    st.markdown("- Sometimes, I hear voices that no one else can hear, and it can be really distressing. I have trouble separating what’s real from what’s not.")
    real_data = st.text_area("Enter a sentence or paragraph for prediction:", 
                              "")
    
    if st.button("Test Prediction"):
        if not real_data.strip():
            st.error("Input cannot be empty! Please enter some text.")
            return 
        processed_real_data = pre_processing([real_data])
        real_data_vectorized = state.vectorizer.transform(processed_real_data).toarray()
        
        additional_features = np.zeros((real_data_vectorized.shape[0], state.data.drop(columns=['post', 'subreddit']).shape[1]))
        
        real_data_features = np.hstack((real_data_vectorized, additional_features))
        
        predicted_class_index = state.model.predict(real_data_features)
        predicted_class = state.label_encoder.inverse_transform(predicted_class_index)
        
        st.write(f"The predicted class for the provided text is: {predicted_class[0]}")

def training():
    if 'training_done' not in state:
        state.training_done = False
    
    st.title("Mental Health Classifier with KNN/Random Forest")
    
    train()  

    if state.training_done:
        st.write("### Test Model with Real Data")
        test_real_data()  
