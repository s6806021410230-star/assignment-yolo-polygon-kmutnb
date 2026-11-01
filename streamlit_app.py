import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("🎯 YOLO Polygon Segmentation WebApp")
st.write("รายวิชา DIP (อ.นพเก้า ทองใบ) — นางสาวสุภักษร สุบินนาม (6806021410230)")

# โหลดโมเดล
@st.cache_resource
def get_model():
    return YOLO("best_polygon.pt")

model = get_model()

# ตัวปรับแต่งค่า
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)
uploaded_file = st.file_uploader("อัปโหลดรูปภาพทดสอบ", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    results = model.predict(source=np.array(img), conf=conf)
    res_plot = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    col1.image(img, caption="ภาพต้นฉบับ")
    col2.image(res_plot, caption="ผลลัพธ์ Polygon Masks")
