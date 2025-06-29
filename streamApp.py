import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np




model = YOLO(r"C:\Users\Archita Shrivastava\Desktop\Custom obj detection\best.pt")

st.set_page_config(page_title = "Vehicle Detection App", layout = "centered")

st.title("Vehicle Detection using YOLOv8")

st.subheader(":red[WELCOME TO MY VEHICLE DETECTION APP]", divider = "violet")

st.markdown('''
            :red[This steamlit app demonstrates real time object detection of images with YOLOv8 model.
            This application is designed to identify and classify multiple types of vehicles such as cars, buses, trucks from images using a custom-trained model.]
            ''')

st.markdown('''
<span style='color: green; font-weight: bold; font-size: 18px;'>KEY FEATURES</span>

- Easy-to-use interface for uploading and testing images.  
- Detection summary table showing class, confidence, and image info.  
- Sidebar with insights and metrics from the training phase.  
- Scalable and customizable for real-world surveillance or traffic analytics use cases.
''', unsafe_allow_html=True)

st.sidebar.title("How the app works")
st.sidebar.info("Yolov8 is used to detect the vehicles in images, the user can upload jpeg, jgp, and png image formats.")

st.sidebar.title("Use Cases")
st.sidebar.info(
    "This app simulates a real-world system that could be used for:\n"
    "1. Smart city traffic surveillance.\n"
    "2. Vehicle flow monitoring.\n"
    "3. Law enforcement detecting over-speeding trucks, etc.\n"
    "4. Highway analytics and toll management."
)
st.sidebar.title("Navigation")
st.sidebar.markdown("## 🚗 Object Detection: Vehicle Types")
st.sidebar.markdown("### 📘 Read More")
st.sidebar.info(
    "This project uses YOLOv8 to detect multiple vehicle types like cars, buses, and trucks. "
    "It was trained on a custom dataset with 12 classes and evaluated with mAP and precision scores."
)


uploaded_file = st.file_uploader("Upload an image", type=["jpg","png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("🔍 Running detections..."):
        results = model.predict(image, conf=0.25)[0]
        result_img = results.plot()

    st.image(result_img, caption="📸 Detected Vehicles", use_column_width=True)

    import pandas as pd

# Extract detected class IDs
detected_classes = results.boxes.cls.cpu().numpy()

# Map class IDs to names
class_counts = {}
for cls in detected_classes:
    cls_name = model.names[int(cls)]
    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

# Format summary table
summary_df = pd.DataFrame(list(class_counts.items()), columns=["Vehicle Type", "Count"])
st.subheader("🚦 Total Detected Vehicles")
st.table(summary_df)

st.subheader("📊 Vehicle Count Bar Chart")

if not summary_df.empty:
    # Convert to bar chart (index = Vehicle Type, values = Count)
    chart_data = summary_df.set_index("Vehicle Type")
    st.bar_chart(chart_data)

