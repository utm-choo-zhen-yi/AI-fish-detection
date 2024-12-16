import streamlit as st
from flask import Flask, request, jsonify
import threading
import os
import cv2
import tempfile
import numpy as np
import csv
from ultralytics import YOLO

# YOLO model path
MODEL_PATH = 'fish_detection_model.pt'
model = YOLO(MODEL_PATH)

# Set up the Flask API
api_app = Flask(__name__)

# Folder to store programmatically uploaded images
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@api_app.route('/upload', methods=['POST'])
def upload_files():
    """API endpoint to handle image uploads."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("file")
    saved_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        saved_files.append(file_path)

    return jsonify({"message": "Files uploaded successfully", "files": saved_files}), 200

# Run Flask in a separate thread
def run_flask():
    api_app.run(host="0.0.0.0", port=8000)

threading.Thread(target=run_flask, daemon=True).start()

# Streamlit app UI
st.title("Fish Detection and Size Analysis")
st.write("Upload multiple images manually or send images programmatically to the `/upload` endpoint.")

# Check for manually uploaded files via Streamlit UI
uploaded_files = st.file_uploader("Upload Images (Manual)", accept_multiple_files=True, type=["jpg", "png"])

# Gather programmatically uploaded files
programmatic_files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f.endswith((".jpg", ".png"))]

if uploaded_files or programmatic_files:
    st.write("Processing images...")
    temp_dir = tempfile.TemporaryDirectory()
    image_paths = []

    # Save Streamlit-uploaded images to a temp directory
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(file_path)

    # Add programmatic uploads
    image_paths.extend(programmatic_files)

    # Find the best image (max fish detected)
    max_fish_count = 0
    best_image_path = None
    for image_path in image_paths:
        image = cv2.imread(image_path)
        results = model(image)
        fish_count = len(results[0].boxes)

        if fish_count > max_fish_count:
            max_fish_count = fish_count
            best_image_path = image_path

    # Process the best image
    if best_image_path:
        st.write(f"Best Image: {os.path.basename(best_image_path)}")
        st.write(f"Number of Fish Detected: {max_fish_count}")

        # Perform detection on the best image
        best_image = cv2.imread(best_image_path)
        results = model(best_image)

        # Calculate fish sizes
        fish_sizes = {}
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, [box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]])
            fish_width = x2 - x1
            fish_height = y2 - y1
            fish_size_px = np.sqrt(fish_width ** 2 + fish_height ** 2)
            fish_size_cm = fish_size_px * 0.05  # Calibration factor (pixels to cm)
            fish_sizes[idx + 1] = fish_size_cm

        # Display results
        st.write("Fish Sizes Detected in Best Image:")
        fish_data = [{"Fish ID": fish_id, "Size (cm)": f"{size:.2f}"} for fish_id, size in fish_sizes.items()]
        st.dataframe(fish_data)

        # Save results to CSV
        csv_output = "fish_size_summary.csv"
        with open(csv_output, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Fish ID", "Size (cm)"])
            for fish_id, size in fish_sizes.items():
                writer.writerow([fish_id, f"{size:.2f}"])

        # Provide download link for CSV
        with open(csv_output, "rb") as f:
            st.download_button(
                label="Download Fish Size Data (CSV)",
                data=f,
                file_name="fish_size_summary.csv",
                mime="text/csv"
            )
    else:
        st.error("No fish detected in any image.")

    # Clean up temporary files
    temp_dir.cleanup()
else:
    st.info("Please upload images or send images programmatically to start processing.")
