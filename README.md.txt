# 📦 AI-Powered Supply Chain Assistant

This Streamlit app predicts late deliveries, explains reasons (using SHAP), and answers natural language questions using LLMs.

## 🚀 Features
- Predicts shipment delay risk
- Explains why (via SHAP)
- Conversational interface (LangChain + OpenAI)
- Optional data retrieval with FAISS

## 🧠 Tech Stack
- Python, Streamlit
- scikit-learn, SHAP
- LangChain, OpenAI API
- FAISS for vector retrieval

## ⚙️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
