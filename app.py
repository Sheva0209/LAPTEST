from ultralytics import YOLO
import streamlit as st
from PIL import Image

# =========================
# LOAD MODEL
# =========================
model = YOLO("runs/detect/train2/weights/best.pt")

# =========================
# UI HEADER
# =========================
st.set_page_config(page_title="Laptop Damage Detection", layout="centered")

st.title("💻 AI Laptop Damage Analyzer")
st.write("Upload gambar laptop untuk mendeteksi kerusakan fisik")

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader(
    "Pilih gambar laptop",
    type=["jpg", "jpeg", "png"]
)

# =========================
# DIAGNOSIS FUNCTION
# =========================
def diagnosis(label):
    label = label.lower()

    if "crack" in label:
        return "Kemungkinan layar rusak, disarankan ganti LCD"
    elif "dent" in label:
        return "Casing penyok, bisa mempengaruhi komponen internal"
    elif "scratch" in label:
        return "Kerusakan ringan, hanya pada permukaan"
    elif "missing" in label:
        return "Komponen hilang, perlu penggantian"
    elif "keyboard" in label:
        return "Keyboard bermasalah atau rusak"
    elif "touchpad" in label:
        return "Touchpad mengalami kerusakan"
    else:
        return "Kerusakan tidak dikenali, perlu pengecekan manual"

# =========================
# MAIN PROCESS
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    with st.spinner("🔍 Menganalisis gambar..."):
        results = model.predict(image)

    st.subheader("📊 Hasil Deteksi:")

    detected = False

    for r in results:
        for box in r.boxes:
            detected = True

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            st.write(f"### 🔧 {label}")
            st.write(f"Confidence: {conf:.2f}")
            st.write(f"Diagnosis: {diagnosis(label)}")

            if conf < 0.6:
                st.warning("⚠️ Deteksi kurang yakin, coba gambar lebih jelas")

    if not detected:
        st.info("Tidak ada kerusakan terdeteksi")

    # =========================
    # SHOW RESULT IMAGE
    # =========================
    result_image = results[0].plot()
    st.image(result_image, caption="Hasil Deteksi AI")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Prototype AI Laptop Damage Detection")