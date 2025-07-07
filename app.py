import streamlit as st
import numpy as np
import joblib


# Load all models and scaler
models = {
    "Linear Regression (Recommended)": joblib.load('linear_regression_model.pkl'),
    "Support Vector Regression (SVR)": joblib.load('best_svr_model.pkl'),
    "XGBoost Regressor": joblib.load('best_xgboost_model.pkl')
}
scaler = joblib.load('scaler.pkl')

# Custom CSS agar mirip dengan contoh gambar dan animasi
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
        animation: fadeInDown 1.2s cubic-bezier(.39,.575,.56,1.000) both;
    }
    .subtitle {
        color: #fff;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        animation: fadeInDown 1.5s cubic-bezier(.39,.575,.56,1.000) both;
    }
    .desc {
        color: #b0b8d1;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 1.5rem;
        animation: fadeIn 2s cubic-bezier(.39,.575,.56,1.000) both;
    }
    .input-card {
        background: #232946;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        padding: 1.5rem 1rem 1rem 1rem;
        margin-bottom: 1.5rem;
        animation: fadeInUp 1.5s cubic-bezier(.39,.575,.56,1.000) both;
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
        animation: fadeInUp 1.5s cubic-bezier(.39,.575,.56,1.000) both, result-glow 2.5s ease-in-out infinite alternate;
        box-shadow: 0 0 0px #2196f3;
        position: relative;
        overflow: hidden;
    }
    @keyframes result-glow {
      0% { box-shadow: 0 0 0px #2196f3; }
      50% { box-shadow: 0 0 24px 6px #2196f3cc; }
      100% { box-shadow: 0 0 0px #2196f3; }
    }
    .result-confetti {
        position: absolute;
        left: 0; right: 0; top: 0; height: 100%; pointer-events: none;
        z-index: 2;
    }
    .confetti {
        position: absolute;
        width: 10px; height: 10px; border-radius: 50%;
        opacity: 0.7;
        animation: confetti-fall 1.8s linear infinite;
    }
    .confetti.c1 { background: #2196f3; left: 20%; animation-delay: 0s; }
    .confetti.c2 { background: #64b5f6; left: 40%; animation-delay: 0.5s; }
    .confetti.c3 { background: #1976d2; left: 60%; animation-delay: 1s; }
    .confetti.c4 { background: #0d47a1; left: 80%; animation-delay: 1.3s; }
    @keyframes confetti-fall {
      0% { top: -15px; opacity: 0.7; }
      80% { opacity: 0.7; }
      100% { top: 100%; opacity: 0; }
    }
    .example-label {
        color: #b0b8d1;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.2rem;
        animation: fadeIn 2s cubic-bezier(.39,.575,.56,1.000) both;
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
    /* Animations */
    @keyframes fadeInDown {
      0% { opacity: 0; transform: translateY(-40px); }
      100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
      0% { opacity: 0; transform: translateY(40px); }
      100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
      0% { opacity: 0; }
      100% { opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">\U0001F4C8 BBCA Saham Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bank Central Asia Stock Prediction</div>', unsafe_allow_html=True)

st.markdown('<div class="desc">Enter the latest stock data to get AI-powered predictions for BBCA\'s future performance</div>', unsafe_allow_html=True)

# Animasi karakter stickman berjalan bolak-balik kiri-kanan + bola loncat di kiri-kanan
st.markdown(
    '''
    <div style="width:100%;height:80px;position:relative;overflow:hidden;margin-bottom:1.5rem;">
      <div class="ball-bounce ball-left"></div>
      <div class="stickman-run-bounce">
        <div class="stickman-walk">
          <div class="head"></div>
          <div class="body"></div>
          <div class="arm left"></div>
          <div class="arm right"></div>
          <div class="leg left"></div>
          <div class="leg right"></div>
        </div>
      </div>
      <div class="ball-bounce ball-right"></div>
    </div>
    <style>
    .ball-bounce {
      position: absolute;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: linear-gradient(135deg, #f7ca18 60%, #fffbe6 100%);
      box-shadow: 0 2px 8px #0a2342aa;
      z-index: 11;
      top: 10px;
      animation: ball-bounce-anim 1.2s cubic-bezier(.68,-0.55,.27,1.55) infinite;
    }
    .ball-left { left: 10px; }
    .ball-right { right: 10px; animation-delay: 0.6s; }
    @keyframes ball-bounce-anim {
      0% { top: 10px; }
      20% { top: 40px; }
      40% { top: 10px; }
      60% { top: 30px; }
      80% { top: 10px; }
      100% { top: 10px; }
    }
    .stickman-run-bounce {
      position: absolute;
      left: 0;
      top: 10px;
      width: 100vw;
      height: 70px; 
      pointer-events: none;
      z-index: 10;
    }
    .stickman-walk {
      position: absolute;
      left: 0;
      top: 10px;
      width: 40px;
      height: 60px;
      animation: stickman-bounce 10s linear infinite;
      transition: transform 0.2s;
    }
    .stickman-walk.reverse {
      transform: scaleX(-1);
    }
    .stickman-walk .head {
      position: absolute;
      left: 10px;
      top: 0px;
      width: 20px;
      height: 20px;
      background: #f7ca18;
      border-radius: 50%;
      border: 2.5px solid #0a2342;
      z-index: 2;
    }
    .stickman-walk .body {
      position: absolute;
      left: 19px;
      top: 18px;
      width: 2.5px;
      height: 22px;
      background: #fff;
      z-index: 1;
    }
    .stickman-walk .arm.left {
      position: absolute;
      left: 19px;
      top: 22px;
      width: 2.5px;
      height: 16px;
      background: #fff;
      transform: rotate(-35deg);
      transform-origin: top left;
      animation: arm-left-move 1s infinite linear;
    }
    .stickman-walk .arm.right {
      position: absolute;
      left: 19px;
      top: 22px;
      width: 2.5px;
      height: 16px;
      background: #fff;
      transform: rotate(35deg);
      transform-origin: top right;
      animation: arm-right-move 1s infinite linear;
    }
    .stickman-walk .leg.left {
      position: absolute;
      left: 19px;
      top: 40px;
      width: 2.5px;
      height: 18px;
      background: #fff;
      transform: rotate(-25deg);
      transform-origin: top left;
      animation: leg-left-move 1s infinite linear;
    }
    .stickman-walk .leg.right {
      position: absolute;
      left: 19px;
      top: 40px;
      width: 2.5px;
      height: 18px;
      background: #fff;
      transform: rotate(25deg);
      transform-origin: top right;
      animation: leg-right-move 1s infinite linear;
    }
    @keyframes stickman-bounce {
      0% { left: 0; }
      45% { left: calc(100vw - 60px); }
      50% { left: calc(100vw - 60px); }
      95% { left: 0; }
      100% { left: 0; }
    }
    @keyframes arm-left-move {
      0% { transform: rotate(-35deg); }
      25% { transform: rotate(-10deg); }
      50% { transform: rotate(-35deg); }
      75% { transform: rotate(-50deg); }
      100% { transform: rotate(-35deg); }
    }
    @keyframes arm-right-move {
      0% { transform: rotate(35deg); }
      25% { transform: rotate(10deg); }
      50% { transform: rotate(35deg); }
      75% { transform: rotate(50deg); }
      100% { transform: rotate(35deg); }
    }
    @keyframes leg-left-move {
      0% { transform: rotate(-25deg); }
      25% { transform: rotate(-50deg); }
      50% { transform: rotate(-25deg); }
      75% { transform: rotate(-10deg); }
      100% { transform: rotate(-25deg); }
    }
    @keyframes leg-right-move {
      0% { transform: rotate(25deg); }
      25% { transform: rotate(50deg); }
      50% { transform: rotate(25deg); }
      75% { transform: rotate(10deg); }
      100% { transform: rotate(25deg); }
    }
    </style>
    <script>
    // Stickman reverse direction on each animation iteration
    setTimeout(function() {
      var stickman = window.parent.document.querySelector('.stickman-walk');
      if (!stickman) return;
      let lastDir = 1;
      stickman.addEventListener('animationiteration', function() {
        lastDir *= -1;
        stickman.classList.toggle('reverse', lastDir === -1);
      });
    }, 1000);
    </script>
    ''', unsafe_allow_html=True)

# Input Card
with st.container():
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('#### <span style="color:#ffffff;">\U0001F4CA Input Stock Data</span>', unsafe_allow_html=True)
    # Model selection
    model_names = list(models.keys())
    model_choice = st.selectbox(
        "Select Model",
        model_names,
        index=0,
        help="Recommended: Linear Regression for best generalization."
    )
    st.markdown('<span style="color:#b0b8d1; font-size:0.95rem;">Recommended: <b>Linear Regression</b> for most stable and interpretable results.</span>', unsafe_allow_html=True)
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
        model_selected = models[model_choice]
        prediction = model_selected.predict(scaled_input)
        st.markdown('''
        <div class="result-card">
            <div class="result-confetti">
                <div class="confetti c1"></div>
                <div class="confetti c2"></div>
                <div class="confetti c3"></div>
                <div class="confetti c4"></div>
            </div>
            Model: <b>{model}</b><br>Close Price Today = <b>Rp {pred:,.2f}</b>
        </div>
        '''.format(model=model_choice, pred=prediction[0]), unsafe_allow_html=True)
else:
    st.markdown('<div class="example-label">\U0001F4CD Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="result-card">Close Price Today = ........</div>', unsafe_allow_html=True)
