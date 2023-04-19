
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Set the device to GPU if available, otherwise to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load the pre-trained BERT model with a single linear classification layer on top
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# Define function for encoding the reviews with BERT tokenizer
def encode_review(review):
    encoded_dict = tokenizer.encode_plus(
                        review,                      
                        add_special_tokens = True, 
                        max_length = 64,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']

# Define function for decoding the sentiment labels
def decode_label(label):
    if label == 1:
        return 'Positive'
    else:
        return 'Negative'

# Load the pre-processed Amazon reviews data into a pandas DataFrame
reviews_df = pd.read_csv('preprocessed_reviews.csv')

# Define the Streamlit app
def app():
    st.title('Sentiment Analysis of Amazon Reviews using BERT')
    review = st.text_input('Enter a review:')
    if st.button('Analyze'):
        # Encode the review using the BERT tokenizer
        input_ids, attention_mask = encode_review(review)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Make a prediction with the fine-tuned BERT model
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            prediction = logits.argmax(dim=1).cpu().item()

        # Decode the predicted sentiment label and display it
        sentiment = decode_label(prediction)
        st.write('The sentiment of the review is:', sentiment)

# Run the Streamlit app
if __name__ == '__main__':
    app()

