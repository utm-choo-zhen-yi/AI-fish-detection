import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import csv
import os
import tempfile

# Load YOLO model
MODEL_PATH = 'fish_detection_model.pt'  # Update this path if needed
model = YOLO(MODEL_PATH)

st.title("Fish Detection and Size Analysis")
st.write("Upload multiple images, and the system will find the image with the most fish, then analyze fish sizes.")

# Upload images
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png"])

if uploaded_files:
    st.write("Processing images...")
    temp_dir = tempfile.TemporaryDirectory()  # Temporary storage for images
    max_fish_count = 0
    best_image_path = None

    # Save uploaded images locally for processing
    image_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(file_path)

    # Find the best image (with maximum fish detected)
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
            fish_size_cm = fish_size_px * 0.05  # Assuming 1 pixel = 0.05 cm calibration factor
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
    st.info("Please upload one or more images to start processing.")
