import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Custom CSS agar mirip dengan contoh gambar
st.set_page_config(page_title="BBCA Saham Prediction", layout="centered")
st.markdown(
    """
    <style>
    .main, .stApp {
        background-color: #181c2b !important;
    }
    .big-title {
        color: #fff;
        background: #0a2342;
        padding: 1.5rem 0 1rem 0;
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        border-radius: 0 0 20px 20px;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #fff;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .desc {
        color: #b0b8d1;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .input-card {
        background: #232946;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        padding: 1.5rem 1rem 1rem 1rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: #0a2342;
        color: #fff;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .example-label {
        color: #b0b8d1;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.2rem;
    }
    label, .stNumberInput label {
        color: #b0b8d1 !important;
        font-weight: 500;
    }
    .stNumberInput input {
        background: #181c2b !important;
        color: #fff !important;
        border-radius: 6px;
        border: 1px solid #232946;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">\U0001F4C8 BBCA Saham Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bank Central Asia Stock Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="desc">Enter the latest stock data to get AI-powered predictions for BBCA\'s future performance</div>', unsafe_allow_html=True)

# Input Card
with st.container():
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('#### <span style="color:#ffffff;">\U0001F4CA Input Stock Data</span>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        open_price = st.number_input("Open Price", value=None, placeholder="ex: 8575", step=100, format="%d")
        high = st.number_input("High", value=None, placeholder="ex: 8675", step=100, format="%d")
        volume = st.number_input("Volume", value=None, placeholder="ex: 68545100", step=100000, format="%d")
    with col2:
        first_trade = st.number_input("First Trade", value=None, placeholder="ex: 8600", step=100, format="%d")
        low = st.number_input("Low", value=None, placeholder="ex: 8575", step=100, format="%d")
    st.markdown('</div>', unsafe_allow_html=True)

# Tombol prediksi
predict_btn = st.button("\U0001F50D Predict Stock Movement", use_container_width=True)

# Hasil prediksi
if predict_btn:
    # Pastikan semua input terisi
    if None in [open_price, first_trade, high, low, volume]:
        st.warning("Please fill in all input fields.")
    else:
        input_data = np.array([[open_price, first_trade, high, low, volume]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        st.markdown(f'<div class="result-card">Close Price Today = <b>Rp {prediction[0]:,.2f}</b></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="example-label">\U0001F4CD Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="result-card">Close Price Today = ........</div>', unsafe_allow_html=True)
