import pandas as pd
import numpy as np
import streamlit as st

# Title of the app
st.title("My First Streamlit App")

# Subheader
st.subheader("Getting started with Streamlit")

# Text
st.write("Hello! 👋 This is my first Streamlit app.")

# Input from user
name = st.text_input("Enter your name:")

# Button
if st.button("Greet Me"):
    st.success(f"Hello {name}, welcome to Streamlit!")

# Slider
age = st.slider("Select your age:", 1, 100, 25)
st.write(f"Your age is {age}")

# Checkbox
if st.checkbox("Show secret message"):
    st.info("Streamlit makes data apps easy 🚀")
