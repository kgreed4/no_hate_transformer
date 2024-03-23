# app.py
import streamlit as st
from transformers import BertTokenizer
from classify import classify_text
import tensorflow as tf

# Set up your Streamlit app
st.title('Text Classification Demo')

# User input
user_input = st.text_area("Enter your text here:")

if st.button('Classify'):
    if user_input:
        loaded_model = tf.saved_model.load('lisan-al-gaib')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        predicted_class = classify_text(loaded_model, tokenizer, user_input)
        if predicted_class == 0:
            st.write('Prediction: HATE')
        elif predicted_class == 1:
            st.write('Prediction: NO HATE')
    else:
        st.write('Please enter some text to classify.')
