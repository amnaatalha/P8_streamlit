import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="CSV Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CSV Data Dashboard")
st.write("Upload a CSV file to generate an interactive dashboard.")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load CSV
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # ---------------------------
    # Data Preview
    # ---------------------------
    st.subheader("👀 Data Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Summary
    # ---------------------------
    st.subheader("📋 Summary Statistics")
    st.write(df.describe(include="all"))

    # ---------------------------
    # KPIs
    # ---------------------------
    st.subheader("📌 Key Metrics")
    num_rows, num_cols = df.shape
    num_missing = df.isnull().sum().sum()
    num_numeric = df.select_dtypes(include=np.number).shape[1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", num_rows)
    col2.metric("Columns", num_cols)
    col3.metric("Missing Values", num_missing)

    # ---------------------------
    # Visualizations
    # ---------------------------
    st.subheader("📈 Visualizations")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Histogram
    if numeric_cols:
        st.markdown("#### Histogram")
        col_choice = st.selectbox("Choose a numeric column for histogram:", numeric_cols, key="hist")
        fig, ax = plt.subplots()
        sns.histplot(df[col_choice].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col_choice}")
        st.pyplot(fig)

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Scatter Plot
    if len(numeric_cols) >= 2:
        st.markdown("#### Scatter Plot")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis:", numeric_cols, index=0, key="x_axis")
        with col2:
            y_axis = st.selectbox("Y-axis:", numeric_cols, index=1, key="y_axis")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
        ax.set_title(f"{y_axis} vs {x_axis}")
        st.pyplot(fig)

    # Box Plot
    if categorical_cols and numeric_cols:
        st.markdown("#### Box Plot")
        col1, col2 = st.columns(2)
        with col1:
            cat_col = st.selectbox("Categorical column:", categorical_cols, key="cat_col")
        with col2:
            num_col = st.selectbox("Numeric column:", numeric_cols, key="num_col")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
        ax.set_title(f"{num_col} by {cat_col}")
        st.pyplot(fig)

else:
    st.info("📂 Please upload a CSV file to see the dashboard.")
