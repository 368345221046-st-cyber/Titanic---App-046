import streamlit as st
import numpy as np
import joblib

# ----------------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="ระบบทำนายผู้รอดชีวิตไททานิค",
    page_icon="🐱",
    layout="centered",
)

# ----------------------------------------------------------------------------
# ค่าคงที่ที่ผู้ใช้ควรแก้ไขให้ตรงกับผลการประเมินโมเดลจริงของตนเอง
# ----------------------------------------------------------------------------
# หมายเหตุ: ค่านี้เป็นค่าตัวอย่างชั่วคราว กรุณาแก้ไขให้ตรงกับผลลัพธ์จริง
# ที่ได้จากการประเมินโมเดล (เช่น accuracy_score บนชุด test) ของคุณ
MODEL_ACCURACY = 0.80  # TODO: แก้ไขเป็นค่าความแม่นยำจริงของโมเดล (0.0 - 1.0)

MODEL_PATH = "titanic_svm.joblib"

# ----------------------------------------------------------------------------
# ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน ใช้แปลงข้อมูลก่อนป้อนเข้าโมเดล (StandardScaler)
# อ้างอิงจากชุดข้อมูล Titanic (train.csv) มาตรฐาน
# หมายเหตุสำคัญ: ไฟล์โมเดลที่ให้มาไม่ได้แนบตัวแปลงข้อมูล (scaler) มาด้วย
# ค่าด้านล่างนี้คำนวณย้อนกลับจาก support vectors ของโมเดลจนตรงกับสถิติ
# ของชุดข้อมูล Titanic ต้นฉบับ หากโมเดลของคุณ fit scaler ด้วยข้อมูลชุดอื่น
# กรุณาแทนที่ด้วยค่า mean/std จาก scaler ที่ใช้ตอนเทรนจริง
# ----------------------------------------------------------------------------
FEATURE_STATS = {
    "Pclass": (2.308642, 0.836071),
    "Sex":    (0.352413, 0.477752),   # เข้ารหัส: หญิง = 1, ชาย = 0
    "Age":    (29.699118, 14.516321),
    "Fare":   (32.204208, 49.665534),
    "SibSp":  (0.523008, 1.102131),
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def scale(value: float, key: str) -> float:
    mean, std = FEATURE_STATS[key]
    return (value - mean) / std


def build_features(pclass, sex_label, age, fare, sibsp) -> np.ndarray:
    sex_value = 1 if sex_label == "หญิง" else 0
    return np.array([[
        scale(pclass, "Pclass"),
        scale(sex_value, "Sex"),
        scale(age, "Age"),
        scale(fare, "Fare"),
        scale(sibsp, "SibSp"),
    ]])


# ----------------------------------------------------------------------------
# สไตล์ CSS แบบมินิมอล
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #FAFBFC; }
    .app-header {
        text-align: center;
        padding: 1.2rem 1rem 0.6rem 1rem;
    }
    .app-header h1 {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1F2933;
        margin-bottom: 0.2rem;
    }
    .app-header p {
        color: #6B7280;
        font-size: 0.95rem;
        margin: 0;
    }
    .mascot {
        font-size: 3.2rem;
        line-height: 1;
    }
    .accuracy-badge {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        text-align: center;
        margin: 0.8rem 0 1.2rem 0;
    }
    .accuracy-badge .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2563EB;
    }
    .accuracy-badge .label {
        font-size: 0.82rem;
        color: #4B5563;
    }
    .result-box {
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        margin-top: 1rem;
    }
    .survive {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
    }
    .not-survive {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
    }
    .footer-dev {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.85rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# ส่วนหัวของหน้าเว็บ
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="mascot">🐱🔔</div>
        <h1>ระบบทำนายผู้รอดชีวิตไททานิค</h1>
        <p>กรอกข้อมูลผู้โดยสารเพื่อทำนายโอกาสรอดชีวิต ด้วยโมเดล Support Vector Machine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="accuracy-badge">
        <div class="value">{MODEL_ACCURACY * 100:.2f}%</div>
        <div class="label">ความแม่นยำของโมเดล (Accuracy) จากการประเมินผล</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# ฟอร์มกรอกข้อมูล
# ----------------------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass_label = st.selectbox(
            "ชั้นโดยสาร (Pclass)",
            options=["ชั้น 1", "ชั้น 2", "ชั้น 3"],
            index=2,
        )
        pclass = {"ชั้น 1": 1, "ชั้น 2": 2, "ชั้น 3": 3}[pclass_label]

        sex_label = st.selectbox("เพศ", options=["ชาย", "หญิง"])

        age = st.number_input(
            "อายุ (ปี)", min_value=0.0, max_value=100.0, value=30.0, step=1.0
        )

    with col2:
        fare = st.number_input(
            "ค่าโดยสาร (Fare, £)", min_value=0.0, max_value=600.0, value=32.0, step=1.0
        )
        sibsp = st.number_input(
            "จำนวนพี่น้อง/คู่สมรสที่โดยสารด้วย (SibSp)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )

    submitted = st.form_submit_button("🔮 ทำนายผล")

if submitted:
    try:
        model = load_model()
        features = build_features(pclass, sex_label, age, fare, sibsp)
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.markdown(
                """
                <div class="result-box survive">
                    <div style="font-size:2.2rem;">🎉🐱</div>
                    <div style="font-size:1.2rem; font-weight:700;">คาดว่า “รอดชีวิต”</div>
                    <div style="font-size:0.9rem; margin-top:0.3rem;">
                        โดราเอมอนดีใจกับคุณด้วย!
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-box not-survive">
                    <div style="font-size:2.2rem;">😿</div>
                    <div style="font-size:1.2rem; font-weight:700;">คาดว่า “ไม่รอดชีวิต”</div>
                    <div style="font-size:0.9rem; margin-top:0.3rem;">
                        ผลลัพธ์นี้เป็นเพียงการคาดการณ์ทางสถิติเท่านั้น
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except FileNotFoundError:
        st.error(
            f"ไม่พบไฟล์โมเดล '{MODEL_PATH}' กรุณาตรวจสอบว่าได้อัปโหลดไฟล์นี้ไว้ในโฟลเดอร์เดียวกับ app.py"
        )
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")

# ----------------------------------------------------------------------------
# ส่วนท้าย - ชื่อผู้พัฒนา
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-dev">
        พัฒนาโดย นายอาทินันท์ วรรณารุณ
    </div>
    """,
    unsafe_allow_html=True,
)
