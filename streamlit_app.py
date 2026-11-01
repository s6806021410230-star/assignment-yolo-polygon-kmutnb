# ==============================================================================
# 🎯 โปรแกรมเว็บแอปพลิเคชัน: YOLO Polygon Segmentation WebApp
# ------------------------------------------------------------------------------
# รายวิชา: 060223218 Digital Image Processing (DIP S1 A-132) — อาจารย์นพเก้า ทองใบ
# ผู้จัดทำ: นางสาวสุภักษร สุบินนาม (รหัสนักศึกษา: 6806021410230 | ที่นั่งสอบ: 17)
# เทคโนโลยี: Streamlit, Ultralytics YOLOv11/v8, OpenCV, Pillow, NumPy
# วัตถุประสงค์: แสดงผลการทำนายโมเดลแบบ Instance Segmentation (Polygon Masks)
# ==============================================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import urllib.request
import io
import time
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# 1. การตั้งค่าหน้าเว็บหลัก (Page Configuration & Metadata)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLO Polygon Segmentation | KMUTNB DIP",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ปรับแต่งหน้าตาของเว็บด้วย Custom CSS ให้สวยงาม ดูเป็นมืออาชีพ
st.markdown("""
<style>
    /* ปรับแต่งหัวข้อหลัก */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 2px;
    }
    /* ปรับแต่งคำอธิบายวิชา */
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    /* ป้ายชื่อนักศึกษาแบบ Gradient */
    .student-badge {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# แสดงหัวข้อหลักและข้อมูลนักศึกษา
st.markdown('<div class="main-title">🎯 YOLO Polygon Segmentation WebApp</div>', unsafe_allow_html=True)
st.markdown("""
<div class="sub-title">
    โครงงานวิชา <b>060223218 Digital Image Processing (DIP S1 A-132)</b> — อาจารย์นพเก้า ทองใบ<br>
    <span class="student-badge">👩‍🎓 นางสาวสุภักษร สุบินนาม | รหัส: 6806021410230 | ที่นั่งสอบ: 17</span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. การโหลดโมเดลพร้อมระบบ Caching (@st.cache_resource)
# ------------------------------------------------------------------------------
# เหตุผลที่ใช้ @st.cache_resource:
# เพื่อให้ Streamlit โหลดโมเดล PyTorch เข้า Memory เพียงครั้งเดียวเท่านั้น
# ป้องกันการโหลดซ้ำทุกครั้งที่ผู้ใช้เลื่อนแถบสไลเดอร์หรือคลิกปุ่ม ทำให้เว็บทำงานรวดเร็วมาก
@st.cache_resource
def load_model():
    # โหลดไฟล์ Weight ที่เทรนเสร็จสมบูรณ์ (best_polygon.pt)
    return YOLO("best_polygon.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดไฟล์โมเดล best_polygon.pt ได้ กรุณาตรวจสอบไฟล์: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 3. แถบควบคุมด้านข้าง (Sidebar Controls & Parameters)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ การตั้งค่าโมเดล (Settings)")
    
    # สไลเดอร์ปรับค่า Confidence Threshold (ความมั่นใจขั้นต่ำของโมเดล)
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="ค่าความมั่นใจขั้นต่ำที่โมเดลต้องมีก่อนจะตัดสินใจตรวจจับวัตถุนั้น"
    )
    
    # สไลเดอร์ปรับค่า IoU Threshold สำหรับ Non-Maximum Suppression (NMS)
    iou_thresh = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="ค่ากำหนดการตัดกรอบและหน้ากากที่ซ้อนทับกันมากเกินไปออก"
    )
    
    st.markdown("---")
    st.subheader("🎨 ตัวเลือกลูกเล่นการแสดงผล (Display Options)")
    # Checkbox สลับเปิด/ปิดการวาด Bounding Box, Mask และ Label ได้อย่างอิสระ
    show_boxes = st.checkbox("ตีกรอบสี่เหลี่ยม (Bounding Boxes)", value=True)
    show_masks = st.checkbox("ระบายสีหน้ากาก (Polygon Masks)", value=True)
    show_labels = st.checkbox("แสดงป้ายชื่อและคะแนน (Labels & Conf)", value=True)
    
    st.markdown("---")
    st.subheader("📋 คลาสที่โมเดลรองรับ (4 Classes)")
    st.markdown("""
    - 👤 **person** (คน)
    - 🚗 **car** (รถยนต์)
    - 🐈 **cat** (แมว)
    - 🐕 **dog** (สุนัข)
    """)
    st.caption("🚀 ขับเคลื่อนด้วยสถาปัตยกรรม YOLOv11 Instance Segmentation")

# ------------------------------------------------------------------------------
# 4. ส่วนการเลือกภาพทดสอบ (Tab 1: Preset Unseen | Tab 2: Upload)
# ------------------------------------------------------------------------------
st.subheader("📥 เลือกรูปภาพทดสอบ (Test Image)")

tab_preset, tab_upload = st.tabs([
    "🌟 เลือกภาพตัวอย่าง Unseen (คลิกทดสอบได้ทันที)",
    "📤 อัปโหลดรูปภาพของตนเอง"
])

selected_img = None

# รายการภาพตัวอย่าง Unseen คุณภาพสูงบนคลาวด์ (โหลดผ่าน URL โดยตรง ใช้งานได้ทันทีบนคลาวด์)
SAMPLE_PRESETS = {
    "🌟 ครบทั้ง 4 คลาสในภาพเดียว (คน, รถยนต์, แมว, สุนัข)": "https://images.unsplash.com/photo-1517423440428-a5a00ad493e8?w=900&q=80",
    "👥 ฝูงคนกลุ่มใหญ่ (Multiple People 9-10 คน)": "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=900&q=80",
    "🚗 รถยนต์และการจราจรกลางเมือง (Cars & Traffic)": "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=900&q=80",
    "🐕 สุนัข 2 ตัววิ่งเล่นบนสนามหญ้า (Two Dogs Running)": "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=900&q=80",
    "🐈 แมวคู่มองออกนอกหน้าต่าง (Two Cats at Window)": "https://images.unsplash.com/photo-1543852786-1cf6624b9987?w=900&q=80"
}

with tab_preset:
    preset_choice = st.selectbox("เลือกซีนทดสอบที่ต้องการ:", list(SAMPLE_PRESETS.keys()))
    if preset_choice:
        try:
            # ดึงข้อมูลไบนารีภาพจาก URL แล้วแปลงเป็น PIL RGB Image
            with urllib.request.urlopen(SAMPLE_PRESETS[preset_choice], timeout=8) as resp:
                selected_img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception:
            st.warning("กำลังดึงภาพตัวอย่างจากเซิร์ฟเวอร์...")

with tab_upload:
    uploaded = st.file_uploader(
        "ลากไฟล์รูปภาพมาวางที่นี่ (รองรับ JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"]
    )
    if uploaded is not None:
        selected_img = Image.open(uploaded).convert("RGB")

# ------------------------------------------------------------------------------
# 5. การทำนายผลด้วยโมเดล YOLO (Model Inference & Visualization)
# ------------------------------------------------------------------------------
if selected_img is not None:
    img_np = np.array(selected_img)
    
    # 5.1 จับเวลาการทำนายผล (Inference Latency Measurement)
    start_time = time.time()
    results = model.predict(
        source=img_np,
        conf=conf_thresh,
        iou=iou_thresh,
        boxes=show_boxes,
        masks=show_masks,
        labels=show_labels,
        verbose=False
    )
    inference_time_ms = (time.time() - start_time) * 1000
    res = results[0]

    # 5.2 เรนเดอร์ผลลัพธ์ภาพที่วาด Mask และ Bounding Box
    annotated_bgr = res.plot(boxes=show_boxes, masks=show_masks, labels=show_labels)
    # แปลงสีจาก BGR (OpenCV) เป็น RGB เพื่อให้สีแสดงผลถูกต้องบนบราวเซอร์
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # 5.3 แสดงภาพเปรียบเทียบ Before / After แบบ 2 คอลัมน์
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 📷 ภาพต้นฉบับ (Original Input)")
        st.image(selected_img, use_container_width=True)
    with col_right:
        st.markdown("#### 🎨 ผลการตัดแบ่งส่วนภาพ (Polygon Segmentation Output)")
        st.image(annotated_pil, use_container_width=True)
        
        # 5.4 ปุ่มดาวน์โหลดภาพผลลัพธ์ลงเครื่อง (In-memory Download Button)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="💾 ดาวน์โหลดรูปผลลัพธ์ (Download Annotated Image)",
            data=buf.getvalue(),
            file_name="yolo_polygon_result.jpg",
            mime="image/jpeg"
        )

    # --------------------------------------------------------------------------
    # 6. แดชบอร์ดสรุปสถิติการตรวจจับ (Detection Analytics Dashboard)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 รายงานสถิติการตรวจจับ (Detection Analytics)")
    
    boxes = res.boxes
    names = model.names
    
    if boxes is not None and len(boxes) > 0:
        # นับจำนวนวัตถุแยกตามคลาส
        counts = {}
        for cls_id in boxes.cls.tolist():
            c_name = names.get(int(cls_id), str(cls_id))
            counts[c_name] = counts.get(c_name, 0) + 1
            
        # สร้างการ์ดสถิติ (Metric Cards)
        metric_cols = st.columns(len(counts) + 2)
        metric_cols[0].metric("⚡ ความเร็วประมวลผล", f"{inference_time_ms:.1f} ms")
        metric_cols[1].metric("🎯 วัตถุที่พบทั้งหมด", f"{len(boxes)} จุด")
        
        icons = {"person": "👤", "car": "🚗", "cat": "🐈", "dog": "🐕"}
        for idx, (c_name, count) in enumerate(counts.items(), 2):
            icon = icons.get(c_name, "📌")
            metric_cols[idx].metric(f"{icon} คลาส {c_name}", f"{count} รายการ")

        # ----------------------------------------------------------------------
        # 7. แกลเลอรีตัดภาพวัตถุเดี่ยว (Cropped Object Gallery)
        # ----------------------------------------------------------------------
        st.markdown("#### 🔍 แกลเลอรีตัดภาพวัตถุที่ตรวจพบ (Cropped Object Gallery)")
        with st.expander("คลิกเพื่อดูวัตถุที่ตรวจพบแยกทีละชิ้นพร้อมคะแนนความมั่นใจ", expanded=True):
            crop_cols = st.columns(min(len(boxes), 5))
            for i, (box, cls_id, conf) in enumerate(zip(boxes.xyxy.cpu().numpy().astype(int), boxes.cls.tolist(), boxes.conf.tolist())):
                x1, y1, x2, y2 = box
                # ตัดเฉพาะส่วนของวัตถุตามพิกัด Bounding Box
                cropped = img_np[max(0, y1):min(img_np.shape[0], y2), max(0, x1):min(img_np.shape[1], x2)]
                c_name = names.get(int(cls_id), str(cls_id))
                col_target = crop_cols[i % len(crop_cols)]
                if cropped.size > 0:
                    col_target.image(
                        cropped,
                        caption=f"#{i+1} {c_name} ({conf*100:.1f}%)",
                        use_container_width=True
                    )
    else:
        st.warning("⚠️ ไม่พบวัตถุตามเกณฑ์ (ลองปรับลดค่า Confidence Threshold ที่แถบด้านซ้ายดูครับ)")

st.markdown("---")
st.caption("🎓 KMUTNB Digital Image Processing Final Assignment | Deployed on Streamlit Community Cloud")
