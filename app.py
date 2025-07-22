import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pandas as pd

st.title("Parking Slot Detection and Analysis")

st.sidebar.header("Upload YOLO Model")
model_file = st.sidebar.file_uploader("Upload YOLO model (.pt)", type=["pt"])

st.sidebar.header("Upload Image")
image_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
        temp_model_file.write(model_file.read())
        model_path = temp_model_file.name
    model = YOLO(model_path)
    st.sidebar.success("Model loaded successfully!")

if image_file and model_file:
    image = Image.open(image_file)
    image_np = np.array(image)
    results = model(image_np, conf=0.25)

    vacant_class_names = ["space-empty", "empty", "vacant", "free", "slot-empty", "empty-slot"]
    occupied_class_names = ["space-occupied", "occupied", "car", "vehicle", "slot-occupied"]

    empty_space_count = sum(
        1 for result in results for cls in result.boxes.cls if model.names[int(cls)].lower() in vacant_class_names
    )
    occupied_space_count = sum(
        1 for result in results for cls in result.boxes.cls if model.names[int(cls)].lower() not in vacant_class_names
    )

    st.subheader("Uploaded Image")
    st.image(image, caption="Original Image", use_container_width=True)
    st.write(f"**Empty Spaces Detected:** {empty_space_count}")
    st.write(f"**Occupied Spaces Detected:** {occupied_space_count}")

    # Detection Results Table
    st.subheader("Detection Results")
    detection_data = []
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            if class_name.lower() in vacant_class_names:
                continue
            detection_data.append({
                "Class": class_name,
                "Confidence": f"{conf:.2f}",
                "Coordinates": f"({x1}, {y1}), ({x2}, {y2})"
            })
    if detection_data:
        st.table(detection_data)
    else:
        st.write("No detections found.")

    # Annotated Image
    annotated_image = image_np.copy()
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            color = (0, 255, 0) if class_name.lower() in vacant_class_names else (255, 0, 0)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_image, f"{class_name} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    st.subheader("Annotated Image")
    st.image(annotated_image, caption="Detections", use_container_width=True)

    # --- Explainability: Heatmaps and Explanations ---
    st.subheader("Explainability: Slot Heatmaps and Explanations")
    explain_data = []
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)].lower()
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            saliency = cv2.Laplacian(gray_crop, cv2.CV_64F)
            heatmap = np.absolute(saliency)
            if heatmap.max() > 0:
                heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
            else:
                heatmap = np.zeros_like(gray_crop)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            density = np.mean(heatmap)
            if class_name in vacant_class_names:
                explanation = "Uniform low-detail region — indicates empty space." if density < 20 else "High variance; possible obstruction."
            else:
                explanation = "High visual complexity — likely due to vehicle presence." if density > 20 else "Distinct edges and shapes — consistent with occupied state."
            # Show in Streamlit
            st.markdown(f"**Slot [{class_name}] ({x1},{y1})-({x2},{y2}) | Confidence: {conf:.2f}**")
            cols = st.columns(3)
            with cols[0]:
                st.image(crop, caption="Slot Crop", use_container_width=True)
            with cols[1]:
                st.image(heatmap_color, caption="Saliency Heatmap", use_container_width=True)
            with cols[2]:
                st.write(explanation)
            explain_data.append({
                'Class': class_name,
                'Box': f"({x1},{y1})-({x2},{y2})",
                'Confidence': round(float(conf), 3),
                'Density': round(float(density), 2),
                'Explanation': explanation
            })
    if explain_data:
        st.subheader("Summary Table")
        st.dataframe(pd.DataFrame(explain_data))
    else:
        st.info("No slot crops/explanations available.")

    # --- SAM Masking Section ---
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    sam_image = image_np.copy()
    if sam_image.shape[2] == 4:
        sam_image = cv2.cvtColor(sam_image, cv2.COLOR_RGBA2RGB)
    sam_masks = mask_generator.generate(sam_image)

    # Define boxes_intersect inline (no utils.py needed)
    def boxes_intersect(boxA, boxB):
        xA1, yA1, xA2, yA2 = boxA
        xB1, yB1, xB2, yB2 = boxB
        return not (xA2 < xB1 or xB2 < xA1 or yA2 < yB1 or yB2 < yA1)

    yolo_boxes = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            yolo_boxes.append((list(map(int, box)), model.names[int(cls)].lower()))

    sam_masked_image = sam_image.copy()
    highlight_image = sam_image.copy()
    for mask in sam_masks:
        mask_box = mask['bbox']
        x1, y1, w, h = mask_box
        x2, y2 = x1 + w, y1 + h
        mask_box_coords = [x1, y1, x2, y2]
        for yolo_box, _ in yolo_boxes:
            if boxes_intersect(mask_box_coords, yolo_box):
                yolo_mask = np.zeros(mask['segmentation'].shape, dtype=np.uint8)
                bx1, by1, bx2, by2 = yolo_box
                yolo_mask[by1:by2, bx1:bx2] = 1
                if np.any(mask['segmentation'] & yolo_mask):
                    color_mask = np.zeros_like(sam_masked_image)
                    color_mask[mask['segmentation']] = [0, 255, 255]
                    sam_masked_image = cv2.addWeighted(sam_masked_image, 1.0, color_mask, 0.4, 0)
                    highlight_mask = np.zeros_like(highlight_image)
                    highlight_mask[mask['segmentation']] = [255, 0, 255]
                    highlight_image = cv2.addWeighted(highlight_image, 1.0, highlight_mask, 0.7, 0)
                    break

    st.subheader("SAM Masked Image")
    st.image(sam_masked_image, caption="SAM Masking", use_container_width=True)
    st.subheader("Highlighted Masked Regions")
    st.image(highlight_image, caption="Highlighted SAM Regions", use_container_width=True)

else:
    st.info("Please upload both a YOLO model and an image to proceed.")
