from GPT2_.inference import generate_text as gpt2_autocomplete
from NGRAM.ngram import autocomplete as ngram_autocomplete
import streamlit as st
import numpy as np

with open("data/en_US.twitter.txt", "r", encoding="utf-8") as file:
    data = file.read()

st.title("Welcome to Autocomplete Website!")

model = st.selectbox(
    'Pilih model autocomplete',
    ('N-Gram', 'GPT-2'))
num_suggestions = st.number_input("Masukkan jumlah suggestions", step=1, value=150)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Generate autocomplete..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if model == "N-Gram":
        response = ngram_autocomplete(prompt, n=2, corpus=data, num_suggestions=num_suggestions)

    else:
        response = gpt2_autocomplete(prompt, num_suggestions)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})