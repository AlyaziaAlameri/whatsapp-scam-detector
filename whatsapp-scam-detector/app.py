import streamlit as st
import pandas as pd
import pickle
import numpy as np

import os

# Load model and vectorizer using pickle with safe paths
model_path = os.path.join(os.path.dirname(__file__), 'scam_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# 1. 🔍 Real-Time Message Input
st.set_page_config(page_title="WhatsApp Scam Detector", layout="wide")
st.title("📱 WhatsApp Scam Detection (UAE Context)")
st.markdown("This tool scans simulated WhatsApp messages and flags potential scams using a trained ML model.")

st.header("🔍 Check a New Message")

user_input = st.text_area("Enter a WhatsApp message to check:", "")

if user_input:
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]
    confidence = model.predict_proba(user_vec)[0][1] * 100

    color = "red" if prediction == 'spam' else "green"
    label = "SPAM" if prediction == 'spam' else "HAM"

    st.markdown(f"""
    <div style='border: 2px solid {color}; padding: 10px; margin-top: 10px; border-radius: 8px'>
        <strong>Prediction:</strong> <span style='color:{color}'>{label} ({confidence:.2f}%)</span>
    </div>
    """, unsafe_allow_html=True)

# 2. 📂 Simulated Inbox CSV
st.header("📬 Scan Simulated WhatsApp Inbox")

# Load simulated inbox
df = pd.read_csv('simulated_inbox.csv')

# Predict
messages = df['message']
X_vec = vectorizer.transform(messages)
predictions = model.predict(X_vec)
probs = model.predict_proba(X_vec)[:, 1]

df['prediction'] = predictions
df['confidence'] = (probs * 100).round(2)

# Filter dropdown
filter_option = st.selectbox("Filter Messages:", ["All", "Only Scam", "Only Safe"])

if filter_option == "Only Scam":
    display_df = df[df['prediction'] == 'spam']
elif filter_option == "Only Safe":
    display_df = df[df['prediction'] == 'ham']
else:
    display_df = df

# Display messages
for _, row in display_df.iterrows():
    color = "red" if row['prediction'] == 'spam' else "green"
    st.markdown(f"""
    <div style='border: 2px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 8px'>
        <strong>From:</strong> {row['sender']}<br>
        <strong>Time:</strong> {row['timestamp']}<br>
        <strong>Message:</strong> {row['message']}<br>
        <strong>Prediction:</strong> <span style='color:{color}'>{row['prediction'].upper()} ({row['confidence']}%)</span>
    </div>
    """, unsafe_allow_html=True)
