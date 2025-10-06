from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
import pandas as pd
import pickle
import os
from rag_utils import load_retriever

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))

template = """
You are a supply chain assistant. The user asked: {question}.
Relevant insights: {insights}.
Answer in a business-friendly and actionable manner.
"""
prompt = PromptTemplate(input_variables=["question", "insights"], template=template)

@st.cache_resource
def load_model_and_data():
    # Load ML model & cleaned dataset (cached)
    model = None
    try:
        with open("supply_model_1.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load model: {e}")

    df_supply_cleaned = pd.read_csv("supply_chain_clean.csv")

    # ✅ Create a unique RecordID since OrderID was dropped
    df_supply_cleaned.reset_index(drop=False, inplace=True)
    df_supply_cleaned.rename(columns={"index": "RecordID"}, inplace=True)

    return model, df_supply_cleaned

@st.cache_resource
def get_cached_retriever(index_dir="faiss_index", k=3):
    return load_retriever(index_dir=index_dir, k=k)

model, df_supply_cleaned = load_model_and_data()

st.title("📦 AI Supply Chain Assistant")
question = st.text_input("Ask me about late deliveries...")

st.subheader("Mode Selection")
mode = st.radio(
    "Select query mode:",
    ["General Q&A (with or without RAG)", "Explain a specific order"],
    index=0
)

use_rag = st.checkbox("Use retrieval-augmented generation (search past shipments)", value=True)
k = st.slider("How many relevant records to retrieve (k)", min_value=1, max_value=10, value=3)

if st.button("Submit"):
    if not question or question.strip() == "":
        st.error("Please type a question first.")
    else:
        # ✅ EXPLANATION MODE
        if mode == "Explain a specific order":
            import joblib
            import shap
            import numpy as np

            # Load precomputed SHAP data
            shap_data = joblib.load("shap_explanations.pkl")
            explainer = shap_data["explainer"]
            X_sample = shap_data["X_sample"]
            shap_values = shap_data["shap_values"]

            # Extract Order ID from query
            import re
            match = re.search(r'\d+', question)
            if match:
                record_id = int(match.group())
            else:
                st.warning("Please include a valid record number in your query, e.g. 'Why is record 452 predicted as late?'")
                st.stop()

            # Check if RecordID exists
            if "RecordID" not in df_supply_cleaned.columns:
                st.error("RecordID column not found in dataset. Please ensure the index reset step ran successfully.")
                st.stop()

            order_row = df_supply_cleaned[df_supply_cleaned["RecordID"] == record_id]
            if order_row.empty:
                st.warning(f"No record found for RecordID {record_id}.")
                st.stop()

            # Get feature importance for this row
            try:
                shap_values_single = explainer.shap_values(order_row.drop(columns=["Late_delivery_risk"]))
            except Exception as e:
                st.error(f"Error generating SHAP explanation: {e}")
                st.stop()

           # Handle both binary and single-output cases safely
            # --- Handle binary / multiclass / regression models gracefully ---
            shap_output = shap_values_single

            # If SHAP returns a list (common for classifiers)
            if isinstance(shap_output, list):
                # Use the last class's SHAP values (usually the "positive" class)
                shap_output = shap_output[-1]

            # Ensure we get a 1D array of SHAP values for this record
            shap_vals_flat = np.ravel(shap_output)  # safely flatten regardless of shape

            # Extract absolute values for ranking
            abs_values = np.abs(shap_vals_flat)

            # Match with feature names
            feature_names = order_row.drop(columns=["Late_delivery_risk"]).columns
            feature_importance = sorted(
                zip(feature_names, abs_values),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            top_features = {f: round(v, 4) for f, v in feature_importance}
            insights = f"Record {record_id} - top factors influencing late delivery: {top_features}"

            user_prompt = prompt.format(question=question, insights=insights)
            response = llm.predict(user_prompt)
            st.write(response)

            st.subheader("🔍 Feature Importance Breakdown")
            st.write(top_features)

            # Optional SHAP bar chart
            shap_df = pd.DataFrame.from_dict(top_features, orient='index', columns=['Importance'])
            st.bar_chart(shap_df)

        # ✅ GENERAL Q&A / RAG MODE
        else:
            if use_rag:
                retriever = get_cached_retriever(index_dir="faiss_index", k=k)
                docs = retriever.get_relevant_documents(question)
                context = "\n".join([d.page_content for d in docs])
                insights = f"Retrieved relevant shipment records:\n{context}"
            else:
                high_risk = df_supply_cleaned[df_supply_cleaned["Late_delivery_risk"] == 1].head(5).to_dict(orient="records")
                insights = f"Found {len(high_risk)} high-risk shipments. Example: {high_risk}"

            user_prompt = prompt.format(question=question, insights=insights)
            response = llm.predict(user_prompt)
            st.write(response)
