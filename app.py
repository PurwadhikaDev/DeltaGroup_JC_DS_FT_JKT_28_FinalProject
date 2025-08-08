import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Prediksi Deposito Bank",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========== LOAD PIPELINE ==========
@st.cache_resource
def load_model():
    return joblib.load('pipeline_voting_model_compressed.pkl')

pipe = load_model()
THRESHOLD = 0.15

# ========== HEADER ==========
st.markdown("<h1 style='text-align: center; color: #005f73;'>💰 Prediksi Penerimaan Deposito Bank</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #0a9396;'>Voting Classifier | SMOTEENN | Preprocessing Pipeline</h4>", unsafe_allow_html=True)
st.write("")
st.markdown(
    "<div style='background-color: #e9d8a6; padding: 10px; border-radius: 10px; text-align:center;'>"
    "Masukkan data calon nasabah di bawah untuk memprediksi kemungkinan menerima penawaran deposito."
    "</div>", unsafe_allow_html=True
)
st.write("")

# ========== INPUT FORM ==========
with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        emp_var_rate = st.number_input("Employment Variation Rate", value=1.1, format="%.2f")
        euribor3m = st.number_input("Euribor 3 Bulan", value=4.56, format="%.2f")
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-39.90, format="%.2f")
        cons_price_idx = st.number_input("Consumer Price Index", value=93.0, format="%.2f")
        age = st.slider("Umur Nasabah", 18, 100, 35)
        education = st.selectbox("Pendidikan", ['illiterate', 'middle_school', 'high_school', 'university', 'professional'])
        generation = st.selectbox("Generasi", ['silent', 'boomers', 'x', 'millenials', 'z', 'alpha'])
        job = st.selectbox("Jenis Pekerjaan", ['admin.', 'blue-collar', 'services', 'technician', 'management', 'entrepreneur', 'retired', 'housemaid', 'self-employed', 'student', 'unemployed'])

    with col2:
        day_of_week = st.selectbox("Hari Kontak", ['mon', 'tue', 'wed', 'thu', 'fri'])
        month = st.selectbox("Bulan Kontak", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        marital = st.selectbox("Status Pernikahan", ['married', 'single', 'divorced', 'unknown'])
        loan = st.selectbox("Status Pinjaman", ['yes', 'no', 'unknown'])
        poutcome = st.selectbox("Outcome Campaign Sebelumnya", ['success', 'failure', 'nonexistent', 'other', 'unknown'])
        contact = st.selectbox("Jenis Kontak", ['cellular', 'telephone', 'unknown'])
        housing = st.selectbox("Status Kredit Rumah", ['yes', 'no', 'unknown'])
        pdays_cat = st.selectbox("Kategori pdays", ['never', 'recent', 'last_month'])

    submitted = st.form_submit_button("🔍 Prediksi")

# ========== PROCESS & OUTPUT ==========
if submitted:
    data_input = pd.DataFrame([{
        'emp.var.rate': emp_var_rate,
        'euribor3m': euribor3m,
        'cons.conf.idx': cons_conf_idx,
        'cons.price.idx': cons_price_idx,
        'age': age,
        'day_of_week': day_of_week,
        'education': education,
        'generation': generation,
        'job': job,
        'month': month,
        'marital': marital,
        'loan': loan,
        'poutcome': poutcome,
        'contact': contact,
        'housing': housing,
        'pdays_cat': pdays_cat
    }])

    probas = pipe.predict_proba(data_input)[:, 1][0]
    pred = int(probas >= THRESHOLD)
    label = "YA (Tertarik Deposito)" if pred == 1 else "TIDAK (Tidak Tertarik)"
    color = "#38b000" if pred == 1 else "#ae2012"
    icon = "✅" if pred == 1 else "❌"

    st.markdown("---")
    st.markdown(f"<h3 style='text-align:center; color:{color};'>{icon} Hasil Prediksi: {label}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size: 20px;'>Probabilitas Tertarik: <b>{probas:.2%}</b></p>", unsafe_allow_html=True)
    st.info(f"Threshold optimal: {THRESHOLD}")

    if pred == 1:
        st.success("🎯 Rekomendasi: **Prioritaskan** nasabah ini dalam campaign marketing!")
    else:
        st.warning("⚠️ Rekomendasi: Peluang kecil, bisa dipertimbangkan untuk tidak diprioritaskan.")

    with st.expander("Lihat detail input nasabah"):
        st.dataframe(data_input.T, use_container_width=True)

# ========== FOOTER ==========
st.markdown("---")
st.caption("🚀 Model: Voting Classifier + SMOTEENN + Preprocessing Pipeline. | Dibangun dengan ❤️ menggunakan Streamlit")

