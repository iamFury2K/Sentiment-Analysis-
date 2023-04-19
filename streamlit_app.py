import streamlit as st
import joblib
from nltk.corpus import stopwords

# Load the pre-trained model and vectorizer
model = joblib.load('mymodel.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Define the app layout
st.title('Sentiment Analysis App')
st.write('Enter some text to predict its sentiment:')
user_input = st.text_input('Text Input')

# Define the app logic
if user_input:
    # Preprocess the input text
    stop_words = stopwords.words('english')
    user_input = ' '.join([word for word in user_input.split() if word not in stop_words])
    
    # Transform the input text using the vectorizer
    user_input_vec = vectorizer.transform([user_input])
    
    # Use the pre-trained model to predict sentiment
    sentiment = model.predict(user_input_vec)[0]
    
    # Display the output
    if sentiment == 1:
        st.write('Sentiment: Positive')
    else:
        st.write('Sentiment: Negative')

