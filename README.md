 Vehicle Detection App with YOLOv8 and Streamlit

This is an end-to-end project that detects multiple types of vehicles (cars, trucks, buses, etc.) in uploaded images using a custom-trained YOLOv8 model. The app is built with Streamlit and provides an interactive interface for users to upload images, view detections, download summaries, and analyze results visually.

 Features

- Real-time vehicle detection using YOLOv8
- Interactive web UI using Streamlit
- Detection summary table with class names, confidence, and bounding boxes
- Class-wise vehicle count + bar chart visualization
- CSV export of detection results

Technologies Used

- Python
- YOLOv8 (Ultralytics)
- Streamlit
- Pandas
- Plotly (optional for charts)
- PIL
 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

