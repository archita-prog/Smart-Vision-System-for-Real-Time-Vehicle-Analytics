Vehicle Detection Web App (YOLOv8 + Streamlit)

An end-to-end machine learning project that detects 12+ types of vehicles (cars, trucks, buses, bikes, etc.) in uploaded images using a custom-trained YOLOv8 model. The application provides an interactive Streamlit interface for uploads , real-time detection 👀, results visualization , and summary export .  

Features

- ✅ Real-time vehicle detection using YOLOv8 (Ultralytics)  
- ✅ Easy-to-use Streamlit web interface 
- ✅ Detection summary table (class name, confidence score, bounding box coordinates)  
- ✅ Class-wise count of detected vehicles (bar chart visualization)  
- ✅ Export detection results as CSV  
- ✅ Works with multiple input images  


Tech Stack

- Python 3.9+  
- YOLOv8 (Ultralytics) → Custom-trained model for vehicle detection  
- Streamlit→ Interactive web app frontend  
- Pandas → Data structuring & CSV export  
- Plotly / Matplotlib → Data visualization (charts, graphs)  
- PIL (Pillow) → Image processing  



Project Structure

```bash
vehicle-detection-app/
│── models/                 # Contains YOLOv8 .pt model file
│── StreamApp.py                  # Main Streamlit app code
│── requirements.txt        # Python dependencies
│── utils.py                # Helper functions (optional: plotting, formatting)
│── sample_data/            # Example test images
│── outputs/                # Saved detection results & exported files
│── README.md               # Project documentation
```



Installation & Setup

1)Clone the Repository
```bash
git clone https://github.com/your-username/vehicle-detection-app.git
cd vehicle-detection-app
```

2)Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3)Install Dependencies
```bash
pip install -r requirements.txt
```

 4) Download / Place YOLOv8 Model
- Place your custom-trained YOLOv8 `.pt` model file in the `models/` folder.  
- Example path: `models/vehicle_yolov8.pt`

***

## ▶️ Running the App

```bash
streamlit run app.py
```

- This will launch the app in your browser (`http://localhost:8501`).  
- Upload an image → View detections → Download CSV results.  


Example Outputs

- Bounding boxes drawn around detected vehicles in the uploaded images  
- A summary table like:  

| Class   | Confidence | Xmin | Ymin | Xmax | Ymax |
|---------|-----------|------|------|------|------|
| Car     | 0.92      | 55   | 103  | 210  | 330  |
| Truck   | 0.87      | 300  | 120  | 500  | 400  |

- A bar chart showing class-wise vehicle counts  



CSV Export

Each run generates a detection summary CSV in the `outputs/` folder containing:  

- Filename  
- Class name  
- Confidence score  
- Bounding box coordinates (Xmin, Ymin, Xmax, Ymax)  



 Customization

- Update `StreamApp.py` to change visualization style (bounding box thickness, colors, charts, etc.).  
- Modify YOLOv8 model path in `streamApp.py` if you want to switch models.  
- Extend to handle video inference or real-time webcam detection (The jupyter notebook is already available for this, check Yolo8 Video Tracker and you can replace it with me webcam also).  



Future Enhancements

- [ ] Add video stream support (dashcams, CCTV)  
- [ ] Deploy app on Streamlit Cloud / HuggingFace Spaces
- [ ] Export detections in **COCO JSON / Pascal VOC XML** formats  



🔥 You’re all set! Now you can detect vehicles with a few clicks in your browser.  



Would you like me to also **include the exact code for `app.py`** (Streamlit implementation with YOLOv8 + Pandas + Plotly) so your README looks complete with example usage?

