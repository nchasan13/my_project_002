import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO  # Import YOLO model from ultralytics
import tkinter as tk  # Import tk for GUI updates
from collections import Counter  # Import Counter for label counting
import os


# Load your YOLO segmentation model (adjust the path if needed)
model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(model_path)


# Method to perform YOLO prediction and display the results
def perform_yolo_prediction(image, display_frame, panel, label_counts_dict, label_count_display, threshold=0.5, nms_threshold=0.45):
    """
    Performs YOLO prediction on the given image and updates the GUI.

    Parameters:
    - image: The input image.
    - display_frame: The frame where the image is displayed.
    - panel: The panel where the image with predictions is shown.
    - label_counts_dict: A dictionary to store label counts.
    - label_count_display: Label widget to display label counts.
    - threshold: Confidence threshold for predictions (default is 0.5).
    - nms_threshold: Non-maximum suppression (NMS) threshold (default is 0.45).
    """
    # Upload image to GPU
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)

    # Resize the image to fit the display frame (use CUDA-based resize)
    resized_image = resize_image(gpu_image, display_frame.winfo_width(), display_frame.winfo_height())

    # Apply YOLO detection using threshold and NMS values
    results = model(resized_image.download(), conf=threshold, iou=nms_threshold)

    # Draw bounding boxes, labels, etc. on the image
    output_image = draw_custom_predictions(resized_image.download(), results[0])

    # Convert the image for Tkinter display
    gpu_rgb = cv2.cuda.cvtColor(cv2.cuda_GpuMat().upload(output_image), cv2.COLOR_BGR2RGB)
    img_rgb = gpu_rgb.download()  # Download the result back to CPU
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # If the panel is None (not yet created), create it
    if panel is None or not panel.winfo_exists():
        panel = tk.Label(display_frame, image=img_tk)
        panel.image = img_tk  # Keep a reference to the image
        panel.grid(row=0, column=0, sticky="nsew")
    else:
        panel.config(image=img_tk)
        panel.image = img_tk  # Keep a reference to the image

    # Update the label counts dictionary
    update_label_counts(results[0], label_counts_dict, label_count_display)

    return output_image  # Optionally return the processed image if needed


# Method to resize image (GPU)
def resize_image(gpu_image, max_width, max_height):
    """
    Resizes an image to fit within the specified width and height while maintaining aspect ratio using GPU.
    """
    img_height, img_width = gpu_image.size()
    aspect_ratio = img_width / img_height

    if max_width / max_height > aspect_ratio:
        display_width = int(max_height * aspect_ratio)
        display_height = max_height
    else:
        display_width = max_width
        display_height = int(max_width / aspect_ratio)

    # Resize the image using CUDA
    resized_img = cv2.cuda.resize(gpu_image, (display_width, display_height))
    return resized_img


# Method to draw custom predictions on the image (still on CPU for display purposes)
def draw_custom_predictions(img, result):
    for box in result.boxes.xyxy:
        # Coordinates of the box
        x1, y1, x2, y2 = map(int, box)
        # Draw the bounding box (thinner, color green)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Thinner box (1 px thickness)
        
    for label, confidence, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
        # Coordinates for text label
        x1, y1, x2, y2 = map(int, box)
        label_text = f"{model.names[int(label)]} {confidence:.2f}"
        # Draw the label text (smaller, white font)
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # Smaller font
        
    return img


# Method to update the label counts
def update_label_counts(result, label_counts_dict, label_count_display):
    """
    Updates the label counts in the given dictionary based on YOLO prediction results.
    """
    # Clear the existing counts in the dictionary
    label_counts_dict.clear()

    # Count the labels from the result
    labels = [model.names[int(label)] for label in result.boxes.cls]
    label_counts = Counter(labels)

    # Update the label counts dictionary
    label_counts_dict.update(label_counts)

    # Calculate total count
    total_count = sum(label_counts_dict.values())

    # Update the label_count_display widget with the new counts
    label_text = "Label Counts:\n\n"
    for label, count in label_counts_dict.items():
        label_text += f"{label}: {count}\n"

    # Add total count to the display
    label_text += f"\nTotal: {total_count}"

    # Update the text in the label_count_display widget
    label_count_display.config(text=label_text)
