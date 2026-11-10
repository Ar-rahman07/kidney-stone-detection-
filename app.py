
import os
import re
import tempfile

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from fpdf import FPDF

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Kidney Stone Detection and Prevention",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CT_MODEL_PATH = os.path.join(
    BASE_DIR, "ct_kidney_stone_model.h5"
)

USG_MODEL_PATH = os.path.join(
    BASE_DIR, "usg_kidney_stone_model.h5"
)

CSV_PATH = os.path.join(
    BASE_DIR, "kidney_stone_urine_3000.csv"
)

CT_DATASET = os.path.join(
    BASE_DIR,
    "datasets",
    "CT SCAN",
    "Kindey Stone Dataset",
    "Augmented"
)

USG_STONE_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "Ultrasound scan",
    "STONE"
)

USG_NO_STONE_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "Ultrasound scan",
    "NO STONE"
)

IMG_SIZE = 128

# Verify these using the class_indices from model training.
CT_STONE_CLASS_INDEX = 1
USG_STONE_CLASS_INDEX = 1

# Match the original training preprocessing.
CT_USE_RGB = True
USG_USE_RGB = True

CT_NORMALIZATION = 255.0
USG_NORMALIZATION = 255.0

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# =========================================================
# 2. CSS
# =========================================================

st.markdown("""
<style>
.stApp {
    background: #2b2f36;
    font-family: Segoe UI;
    color: white;
}

h1 {
    color: white !important;
    text-align: center;
    font-weight: 900;
}

h2, h3, h4, p, label {
    color: white !important;
}

.card {
    background: #1f2933;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 20px;
}

.info {
    background: #1e293b;
    color: #e5e7eb;
    padding: 14px;
    border-left: 6px solid #38bdf8;
    border-radius: 8px;
    margin-top: 12px;
}

.success {
    background: #1f3d2b;
    color: #ffffff;
    padding: 14px;
    border-left: 6px solid #22c55e;
    border-radius: 8px;
    margin-top: 12px;
}

.tip-card {
    background: #303944;
    color: #ffffff !important;
    padding: 13px;
    margin-top: 8px;
    border-radius: 7px;
    font-size: 16px;
}

.stTextInput input,
.stNumberInput input {
    background-color: #f8fafc !important;
    color: #000000 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div {
    background-color: #f8fafc !important;
    color: #000000 !important;
}

div[data-baseweb="select"] span {
    color: #000000 !important;
}

.stButton button {
    background: #0284c7;
    color: white;
    border-radius: 8px;
    width: 100%;
    height: 45px;
    font-weight: bold;
}

.stDownloadButton button {
    background: #15803d;
    color: white;
    border-radius: 8px;
    width: 100%;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    ct = tf.keras.models.load_model(
        CT_MODEL_PATH
    )

    usg = tf.keras.models.load_model(
        USG_MODEL_PATH
    )

    return ct, usg


try:
    ct_model, usg_model = load_models()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# =========================================================
# 4. IMAGE PREPROCESSING
# =========================================================

def preprocess_image(img, model, scan_type):

    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    height = shape[1] or IMG_SIZE
    width = shape[2] or IMG_SIZE
    channels = shape[3]

    resized = cv2.resize(
        img,
        (int(width), int(height))
    )

    if scan_type == "CT Scan":
        use_rgb = CT_USE_RGB
        normalization = CT_NORMALIZATION
    else:
        use_rgb = USG_USE_RGB
        normalization = USG_NORMALIZATION

    if channels == 1:
        resized = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY
        )
        resized = np.expand_dims(
            resized,
            axis=-1
        )

    elif channels == 3 and use_rgb:
        resized = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

    resized = resized.astype("float32")

    if normalization:
        resized /= normalization

    return np.expand_dims(resized, axis=0)


# =========================================================
# 5. CNN PREDICTION
# =========================================================

def predict_scan(model, img, scan_type):

    x = preprocess_image(
        img,
        model,
        scan_type
    )

    output = np.asarray(
        model.predict(x, verbose=0)
    )

    if output.ndim != 2:
        raise ValueError(
            "Unsupported model output."
        )

    if scan_type == "CT Scan":
        stone_index = CT_STONE_CLASS_INDEX
    else:
        stone_index = USG_STONE_CLASS_INDEX

    if output.shape[-1] == 1:

        raw = float(output[0][0])

        if stone_index == 1:
            probability = raw
        else:
            probability = 1 - raw

    elif output.shape[-1] == 2:

        probability = float(
            output[0][stone_index]
        )

    else:
        raise ValueError(
            "Model must have one sigmoid or two softmax outputs."
        )

    prediction = (
        "YES" if probability >= 0.5 else "NO"
    )

    confidence = max(
        probability,
        1 - probability
    ) * 100

    return (
        prediction,
        round(confidence, 2)
    )


# =========================================================
# 6. DATASET IMAGE COLLECTION
# =========================================================

def collect_images(dataset_path, positive_folder,
                   negative_folder=None):

    records = []

    if negative_folder is not None:

        folders = [
            (positive_folder, 1),
            (negative_folder, 0)
        ]

    else:

        folders = [
            (os.path.join(dataset_path, "Stone"), 1),
            (os.path.join(dataset_path, "Non-Stone"), 0)
        ]

    for folder, label in folders:

        if not os.path.isdir(folder):
            continue

        for root, _, files in os.walk(folder):

            for filename in files:

                if filename.lower().endswith(
                    IMAGE_EXTENSIONS
                ):

                    path = os.path.join(
                        root,
                        filename
                    )

                    # Extract patient ID where available.
                    match = re.search(
                        r"(P\d+)",
                        filename,
                        re.IGNORECASE
                    )

                    group = (
                        match.group(1).upper()
                        if match
                        else os.path.basename(path)
                    )

                    records.append(
                        (path, label, group)
                    )

    return records


# =========================================================
# 7. IMAGE MODEL ACCURACY
# =========================================================

def calculate_image_accuracy(
    records,
    model,
    scan_type
):

    if len(records) < 10:
        return None

    df = pd.DataFrame(
        records,
        columns=["path", "label", "group"]
    )

    # Keep patient groups separate when possible.
    unique_groups = df["group"].nunique()

    if unique_groups >= 5:

        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=42
        )

        _, test_indices = next(
            splitter.split(
                df,
                df["label"],
                groups=df["group"]
            )
        )

        test_df = df.iloc[test_indices]

    else:

        _, test_df = train_test_split(
            df,
            test_size=0.20,
            random_state=42,
            stratify=df["label"]
        )

    actual = []
    predicted = []

    for _, row in test_df.iterrows():

        image = cv2.imread(
            row["path"]
        )

        if image is None:
            continue

        prediction, _ = predict_scan(
            model,
            image,
            scan_type
        )

        actual.append(
            int(row["label"])
        )

        predicted.append(
            1 if prediction == "YES" else 0
        )

    if len(actual) == 0:
        return None

    return round(
        accuracy_score(actual, predicted) * 100,
        2
    )


# =========================================================
# 8. CALCULATE CT AND USG ACCURACY
# =========================================================

@st.cache_resource
def get_scan_accuracies():

    ct_records = collect_images(
        CT_DATASET,
        None
    )

    usg_records = collect_images(
        "",
        USG_STONE_DIR,
        USG_NO_STONE_DIR
    )

    ct_accuracy = calculate_image_accuracy(
        ct_records,
        ct_model,
        "CT Scan"
    )

    usg_accuracy = calculate_image_accuracy(
        usg_records,
        usg_model,
        "Ultrasound Scan"
    )

    return ct_accuracy, usg_accuracy


with st.spinner("Calculating model accuracy..."):

    try:
        CT_ACC, USG_ACC = get_scan_accuracies()

    except Exception:
        CT_ACC, USG_ACC = None, None


# =========================================================
# 9. IMAGE HIGHLIGHTING
# =========================================================

def detect_and_circle(img, scan_type):

    result = img.copy()

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    found = False
    pixels = 0

    if scan_type == "CT Scan":

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        _, threshold = cv2.threshold(
            blur,
            240,
            255,
            cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        valid = [
            c for c in contours
            if cv2.contourArea(c) > 8
        ]

        if valid:

            contour = max(
                valid,
                key=cv2.contourArea
            )

            x, y, cw, ch = cv2.boundingRect(
                contour
            )

            cx = x + cw // 2
            cy = y + ch // 2

            radius = max(
                5,
                max(cw, ch) // 2
            )

            cv2.circle(
                result,
                (cx, cy),
                radius,
                (255, 0, 0),
                2
            )

            cv2.rectangle(
                result,
                (x, y),
                (x + cw, y + ch),
                (0, 255, 0),
                2
            )

            pixels = max(cw, ch)
            found = True

    else:

        roi = gray[
            int(0.2 * h):int(0.8 * h),
            int(0.2 * w):int(0.8 * w)
        ]

        if roi.size > 0:

            blur = cv2.GaussianBlur(
                roi,
                (5, 5),
                0
            )

            _, threshold = cv2.threshold(
                blur,
                180,
                255,
                cv2.THRESH_BINARY
            )

            contours, _ = cv2.findContours(
                threshold,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            valid = [
                c for c in contours
                if cv2.contourArea(c) > 8
            ]

            if valid:

                contour = max(
                    valid,
                    key=cv2.contourArea
                )

                x, y, cw, ch = cv2.boundingRect(
                    contour
                )

                x += int(0.2 * w)
                y += int(0.2 * h)

                cx = x + cw // 2
                cy = y + ch // 2

                radius = max(
                    5,
                    max(cw, ch) // 2
                )

                cv2.circle(
                    result,
                    (cx, cy),
                    radius,
                    (255, 0, 0),
                    2
                )

                cv2.rectangle(
                    result,
                    (x, y),
                    (x + cw, y + ch),
                    (0, 255, 0),
                    2
                )

                pixels = max(cw, ch)
                found = True

    return result, pixels, found


# =========================================================
# 10. URINE ML MODEL
# =========================================================

@st.cache_resource
def train_urine_model():

    df = pd.read_csv(CSV_PATH)

    df.columns = (
        df.columns.str.strip().str.lower()
    )

    required = [
        "gravity",
        "ph",
        "urea",
        "calc",
        "osmo",
        "target"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}"
        )

    df = df.dropna(
        subset=required
    )

    X = df[
        ["gravity", "ph", "urea", "calc", "osmo"]
    ]

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = round(
        accuracy_score(y_test, predictions) * 100,
        2
    )

    return model, scaler, accuracy


try:

    ml, scaler, URINE_ML_ACCURACY = train_urine_model()

except Exception as e:

    st.error(f"Urine model error: {e}")
    st.stop()


# =========================================================
# 11. URINE PREDICTION
# =========================================================

def urine_ml(g, ph_val, u, c, o):

    X = pd.DataFrame(
        [[g, ph_val, u, c, o]],
        columns=[
            "gravity",
            "ph",
            "urea",
            "calc",
            "osmo"
        ]
    )

    scaled = scaler.transform(X)

    prediction = ml.predict(scaled)[0]

    probabilities = ml.predict_proba(scaled)[0]

    positive_index = list(
        ml.classes_
    ).index(1)

    probability = float(
        probabilities[positive_index]
    )

    result = (
        "YES" if prediction == 1 else "NO"
    )

    return result, probability


# =========================================================
# 12. HYDRATION
# =========================================================

def hydration(g):

    if g > 1.020:
        return "Dehydration"

    elif g < 1.010:
        return "Over Hydration"

    return "Normal Hydration"


# =========================================================
# 13. STONE INFORMATION
# =========================================================

def stone_info(ph_val, c, u, o):

    if c > 3.5 and ph_val <= 7:
        return (
            "Calcium Oxalate",
            "Calcium-based",
            "High recurrence risk"
        )

    if ph_val < 5.5 and u > 400:
        return (
            "Uric Acid",
            "Metabolic",
            "Possible if acidic urine"
        )

    if o > 700:
        return (
            "Mixed Stone",
            "Concentration-related",
            "Possible"
        )

    return (
        "Low Risk / Unknown",
        "N/A",
        "Low possibility"
    )


# =========================================================
# 14. RISK LEVEL
# =========================================================

def get_risk_level(prob):

    if prob >= 0.75:
        return "High Risk"

    elif prob >= 0.45:
        return "Moderate Risk"

    return "Low Risk"


# =========================================================
# 15. PREVENTION TIPS
# =========================================================

def get_prevention_tips(
    hyd,
    stone_name,
    urine_prob,
    stone_detected
):

    tips = []

    if hyd == "Dehydration":

        tips.append(
            "Maintain adequate daily fluid intake as advised by your doctor."
        )

    elif hyd == "Over Hydration":

        tips.append(
            "Maintain balanced water intake and avoid excessive fluid intake."
        )

    else:

        tips.append(
            "Maintain proper hydration daily."
        )

    if stone_name == "Calcium Oxalate":

        tips.append(
            "Discuss oxalate-rich foods and dietary calcium with your doctor."
        )

        tips.append(
            "Limit excess salt intake."
        )

    elif stone_name == "Uric Acid":

        tips.append(
            "Discuss red meat and high-purine food intake with your doctor."
        )

        tips.append(
            "Ask your healthcare provider about urine pH management."
        )

    elif stone_name == "Mixed Stone":

        tips.append(
            "Monitor salt, protein and fluid intake with professional advice."
        )

    if urine_prob >= 0.75:

        tips.append(
            "Medical consultation is recommended for further evaluation."
        )

    if stone_detected == "YES":

        tips.append(
            "Consult a doctor for review of the scan result."
        )

    if not tips:

        tips.append(
            "Continue healthy diet and hydration habits."
        )

    return tips


# =========================================================
# 16. PDF REPORT - TWO PAGES
# =========================================================

def create_pdf(
    name, age, sex, scan_type,
    detected, confidence, stone_size,
    orig, det_img,
    gravity, ph, urea, calcium, osmo,
    hyd, urine_risk, urine_prob,
    risk_level, sname, scat, sfuture,
    prevention_tips
):

    pdf = FPDF(
        format="A4"
    )

    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(False)

    margin = 15
    content_width = pdf.w - 30

    def line(text, bold=False, size=10, height=7):

        pdf.set_x(margin)

        pdf.set_font(
            "Arial",
            "B" if bold else "",
            size
        )

        pdf.multi_cell(
            content_width,
            height,
            str(text)
        )

    original_path = None
    detected_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as f:
            original_path = f.name

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as f:
            detected_path = f.name

        cv2.imwrite(original_path, orig)
        cv2.imwrite(detected_path, det_img)

        # PAGE 1

        pdf.add_page()

        pdf.set_font("Arial", "B", 17)

        pdf.set_xy(margin, 15)

        pdf.cell(
            content_width,
            12,
            "Kidney Stone Detection Report",
            align="C"
        )

        pdf.set_y(32)

        line("PATIENT INFORMATION", True)
        line(f"Patient Name: {name}")
        line(f"Age: {age}")
        line(f"Sex: {sex}")
        line(f"Scan Type: {scan_type}")

        pdf.ln(2)

        line("PREDICTION RESULT", True)
        line(f"Stone Detected: {detected}")
        line(f"Detection Confidence: {confidence}%")
        line(f"Stone Size: {stone_size}")

        pdf.set_xy(margin, 112)
        line("UPLOADED IMAGE", True)

        pdf.image(
            original_path,
            x=55,
            y=122,
            w=100,
            h=55
        )

        pdf.set_xy(margin, 181)
        line("HIGHLIGHTED IMAGE", True)

        pdf.image(
            detected_path,
            x=55,
            y=191,
            w=100,
            h=55
        )

        # PAGE 2

        pdf.add_page()

        pdf.set_font("Arial", "B", 17)

        pdf.set_xy(margin, 15)

        pdf.cell(
            content_width,
            12,
            "Urine Test Analysis",
            align="C"
        )

        pdf.set_y(32)

        line("URINE ANALYSIS", True)

        details = [
            f"Specific Gravity: {gravity}",
            f"pH: {ph}",
            f"Urea: {urea}",
            f"Calcium: {calcium}",
            f"Osmolality: {osmo}",
            f"Hydration: {hyd}",
            f"Stone Risk (ML): {urine_risk}",
            f"Urine Risk Probability: {urine_prob * 100:.2f}%",
            f"Risk Level: {risk_level}",
            f"Stone Name: {sname}",
            f"Category: {scat}",
            f"Future Possibility: {sfuture}"
        ]

        for item in details:
            line(item)

        pdf.ln(3)

        line("PREVENTION TIPS", True)

        for tip in prevention_tips:
            line("- " + tip)

        pdf.ln(3)

        pdf.set_font("Arial", "I", 9)

        line(
            "Note: This is an experimental ML decision-support "
            "application, not a medical diagnosis. Consult a doctor.",
            height=6
        )

        return bytes(pdf.output())

    finally:

        if original_path and os.path.exists(original_path):
            os.remove(original_path)

        if detected_path and os.path.exists(detected_path):
            os.remove(detected_path)


# =========================================================
# 17. MAIN APPLICATION
# =========================================================

st.title(
    "Kidney Stone Detection and Prevention"
)

# PATIENT DETAILS

st.markdown(
    "<div class='card'>",
    unsafe_allow_html=True
)

st.subheader("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("Patient Name")

with col2:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=25
    )

with col3:
    sex = st.selectbox(
        "Sex",
        ["Male", "Female", "Other"]
    )

st.markdown("</div>", unsafe_allow_html=True)


# SCAN INPUT

st.markdown(
    "<div class='card'>",
    unsafe_allow_html=True
)

st.subheader("Scan Image")

scan_type = st.selectbox(
    "Scan Type",
    ["CT Scan", "Ultrasound Scan"]
)

file = st.file_uploader(
    "Upload Scan Image",
    type=["jpg", "jpeg", "png"]
)

st.markdown("</div>", unsafe_allow_html=True)


# URINE INPUT

st.markdown(
    "<div class='card'>",
    unsafe_allow_html=True
)

st.subheader("Urine Analysis")

col1, col2 = st.columns(2)

with col1:

    gravity = st.number_input(
        "Specific Gravity",
        min_value=1.000,
        max_value=1.050,
        value=1.015,
        step=0.001,
        format="%.3f"
    )

    ph = st.number_input(
        "pH",
        min_value=4.0,
        max_value=9.0,
        value=6.5
    )

    urea = st.number_input(
        "Urea",
        value=300.0
    )

with col2:

    calcium = st.number_input(
        "Calcium",
        value=2.5
    )

    osmo = st.number_input(
        "Osmolality",
        value=500.0
    )

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 18. GENERATE RESULT
# =========================================================

if st.button("Generate Result"):

    if file is None:

        st.error("Please upload a scan image.")

    else:

        img = cv2.imdecode(
            np.frombuffer(file.getvalue(), np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:

            st.error("Invalid image file.")

        else:

            try:

                orig = img.copy()

                if scan_type == "CT Scan":
                    model = ct_model
                else:
                    model = usg_model

                detected, confidence = predict_scan(
                    model,
                    img,
                    scan_type
                )

                det_img, stone_pixels, stone_found = (
                    detect_and_circle(
                        img.copy(),
                        scan_type
                    )
                )

                # Show pixel size only; not a calibrated mm measurement.
                stone_size = (
                    f"{stone_pixels} pixels"
                    if stone_found
                    else "Not detected"
                )

                urine_risk, urine_prob = urine_ml(
                    gravity,
                    ph,
                    urea,
                    calcium,
                    osmo
                )

                risk_level = get_risk_level(
                    urine_prob
                )

                hyd = hydration(gravity)

                sname, scat, sfuture = stone_info(
                    ph,
                    calcium,
                    urea,
                    osmo
                )

                prevention_tips = get_prevention_tips(
                    hyd,
                    sname,
                    urine_prob,
                    detected
                )

                # IMAGE DISPLAY

                st.subheader("Scan Image Analysis")

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        cv2.cvtColor(
                            orig,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Uploaded Image",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        cv2.cvtColor(
                            det_img,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Stone Highlighted Image",
                        use_container_width=True
                    )

                # RESULT

                st.markdown(
                    "<div class='info'><b>PREDICTION RESULT</b></div>",
                    unsafe_allow_html=True
                )

                st.write("**Stone Detected:**", detected)

                st.write(
                    "**Detection Confidence:**",
                    f"{confidence}%"
                )

                st.write(
                    "**Stone Size:**",
                    stone_size
                )

                st.write("**Hydration:**", hyd)

                st.write(
                    "**Stone Risk (ML):**",
                    urine_risk
                )

                st.write(
                    "**Urine Risk Probability:**",
                    f"{urine_prob * 100:.2f}%"
                )

                st.write(
                    "**Risk Level:**",
                    risk_level
                )

                st.write("**Stone Name:**", sname)

                st.write("**Category:**", scat)

                st.write(
                    "**Future Possibility:**",
                    sfuture
                )

                # MODEL ACCURACY

                st.markdown(
                    "<div class='info'><b>MODEL ACCURACY</b></div>",
                    unsafe_allow_html=True
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "CT Scan CNN Accuracy",
                        f"{CT_ACC:.2f}%" if CT_ACC is not None else "Unavailable"
                    )

                with col2:

                    st.metric(
                        "Ultrasound CNN Accuracy",
                        f"{USG_ACC:.2f}%" if USG_ACC is not None else "Unavailable"
                    )

                with col3:

                    st.metric(
                        "Urine ML Accuracy",
                        f"{URINE_ML_ACCURACY:.2f}%"
                    )

                # PREVENTION TIPS

                st.markdown(
                    """
                    <div class="success">
                        <h3 style="color:white;">
                            PREVENTION TIPS
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                for tip in prevention_tips:

                    st.markdown(
                        f"""
                        <div class="tip-card">
                            &#8226;&nbsp; {tip}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # PDF

                pdf_data = create_pdf(
                    name,
                    age,
                    sex,
                    scan_type,
                    detected,
                    confidence,
                    stone_size,
                    orig,
                    det_img,
                    gravity,
                    ph,
                    urea,
                    calcium,
                    osmo,
                    hyd,
                    urine_risk,
                    urine_prob,
                    risk_level,
                    sname,
                    scat,
                    sfuture,
                    prevention_tips
                )

                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name="Kidney_Stone_Report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:

                st.error(f"Error: {e}")


# =========================================================
# 19. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Kidney Stone Detection and Prevention | "
    "Machine Learning Project"
)
