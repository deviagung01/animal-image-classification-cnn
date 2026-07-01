import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# =====================================
# KONFIGURASI HALAMAN
# =====================================
st.set_page_config(
    page_title="Animal Classification AI",
    page_icon="🐾",
    layout="wide"
)

# =====================================
# CSS MODERN
# =====================================
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a,
        #3b82f6
    );
}

[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}

.block-container{
    padding-top: 2rem;
}

.title{
    text-align:center;
    color:white;
    font-size:55px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:20px;
    margin-bottom:30px;
}

.prediksi{
    background:white;
    padding:25px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 5px 15px rgba(0,0,0,0.3);
}

.info-card{
    background:rgba(255,255,255,0.15);
    padding:20px;
    border-radius:15px;
    color:white;
    margin-bottom:20px;
}

/* EXPANDER */
[data-testid="stExpander"] summary {
    color: white !important;
}

[data-testid="stExpander"] details summary p {
    color: white !important;
    font-size: 18px !important;
    font-weight: bold !important;
}

[data-testid="stExpander"]{
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown(
    '<div class="subtitle">Sistem Klasifikasi Gambar Hewan Menggunakan Convolutional Neural Network (CNN)</div>',
    unsafe_allow_html=True
)

# =====================================
# LOAD MODEL
# =====================================
model = load_model("model_hewan.h5")

# =====================================
# LOAD CLASS
# =====================================
classes = np.load(
    "classes.npy",
    allow_pickle=True
)

IMG_SIZE = 150

# =====================================
# INFORMASI SISTEM
# =====================================
st.markdown("""
<div class="info-card">
<h3>📌 Informasi Sistem</h3>

• Metode : Convolutional Neural Network (CNN)<br>
• Jumlah Kelas : 10 Hewan<br>
• Input Gambar : 150 x 150 Pixel<br>
• Dataset : Dataset Hewan<br>

</div>
""", unsafe_allow_html=True)

# =====================================
# UPLOAD GAMBAR
# =====================================
uploaded_file = st.file_uploader(
    "📤 Upload Gambar Hewan",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDIKSI
# =====================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Tampilkan gambar
    st.markdown(
        "<h4 style='color:white;text-align:center'>📷 Gambar yang Diupload</h4>",
        unsafe_allow_html=True
    )

    # =====================================
    # PREPROCESSING
    # =====================================
    img = np.array(image)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    # =====================================
    # PREDIKSI
    # =====================================
    pred = model.predict(img)

    class_idx = np.argmax(pred)

    confidence = np.max(pred) * 100

    # =====================================
    # TAMPILAN HASIL
    # =====================================
    col1, col2 = st.columns([2, 1])

    with col1:

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="prediksi">
                <h2>🐾 {classes[class_idx].upper()}</h2>
                <h3>{confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(
            int(confidence)
        )

    # =====================================
    # DETAIL PROBABILITAS
    # =====================================
    with st.expander("📊 Detail Probabilitas Setiap Kelas"):

        for i, kelas in enumerate(classes):

            persen = float(pred[0][i] * 100)

            st.markdown(
                f"<h5 style='color:white'>{kelas}</h5>",
                unsafe_allow_html=True
            )

            st.progress(
                int(persen)
            )

            st.markdown(
                f"<p style='color:white'>{persen:.2f}%</p>",
                unsafe_allow_html=True
            )

# =====================================
# FOOTER
# =====================================
st.markdown("""
<br><br>

<center style='color:white'>
Dibuat menggunakan Python • Streamlit • TensorFlow • CNN
</center>
""", unsafe_allow_html=True)