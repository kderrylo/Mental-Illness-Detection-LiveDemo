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


def training():
    params = {}

    model_option = st.selectbox("Model:", list(model_options.keys()))
    params['test_size'] = st.slider("Test Dataset Size", 0.05, 0.9, 0.2)
    params['random_state'] = st.number_input("Random State", 0, 100, 42)

    params = dict(params, **dict([(k, st.number_input(k, min_value=r[0], max_value=r[1], value=r[2])) for k, r in param_options.items()
                                  if k in model_options[model_option].__init__.__code__.co_varnames]))

    data = pd.read_csv('Preprocessed.csv')
    data.columns = ['prepro']
    df = pd.read_csv(
        'https://raw.githubusercontent.com/kderrylo/Mental-Illness-detection-using-Sentiment-Analysis/refs/heads/master/mental_health_pre%2Bpost_pandemic_reddit.csv')
    df = pd.concat([df, data], axis=1)
    st.write(pd.DataFrame(df))
    posts = df['prepro']
    labels = df['subreddit']

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(posts)
    y = labels

    # Trigger training process on button click
    if st.button("Train"):
        state.training_done = False
        model_params = model_options[model_option].__init__.__code__.co_varnames

        # Assuming data is already loaded into session state
        state.x_train, state.x_test, state.y_train, state.y_test = train_test_split(
            X, y, test_size=params['test_size'], random_state=params['random_state'])

        # Train the selected model
        model_params = {k: v for k,
                        v in params.items() if k in model_params}
        state.model = model_options[model_option](
            **model_params).fit(state.x_train, state.y_train)

        # Make predictions and evaluate the model
        y_pred = state.model.predict(state.x_test)

        # Calculate classification metrics
        accuracy = accuracy_score(state.y_test, y_pred)
        precision = precision_score(
            state.y_test, y_pred, average='weighted')
        recall = recall_score(state.y_test, y_pred, average='weighted')
        f1 = f1_score(state.y_test, y_pred, average='weighted')

        label_encoder = LabelEncoder()
        state.y_train = label_encoder.fit_transform(state.y_train)
        state.y_test = label_encoder.transform(state.y_test)

        # Make predictions
        y_pred = state.model.predict(state.x_test)
        y_pred = label_encoder.transform(y_pred)
        mse = mean_squared_error(state.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(state.y_test, y_pred)
        r2 = r2_score(state.y_test, y_pred)

        report = classification_report(
            state.y_test, y_pred, target_names=state.model.classes_)

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

        with open(f'{model_option.lower()}_model.pkl', 'wb') as mo:
            pickle.dump(state.model, mo)

    return