import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, IntVar
from PIL import Image, ImageTk
import os
import cv2  # Import cv2 to handle image reading
from collections import Counter
import yt_dlp  # For downloading YouTube videos
import multiprocessing

# Import methods from methods.py
from methods_02 import perform_yolo_prediction

# Global variables for video processing
video_cap = None  # VideoCapture object
frame_rate = 25  # Default frame rate for video (can be adjusted based on the video)
current_frame = None
playing = False  # Flag to control video playback
panel = None  # Panel for video display
output_video_writer = None  # For saving processed video
save_video_flag = False  # Flag for saving video or not
label_counts_dict = {}  # Reference to the label count text widget


def open_video_file():
    global video_cap, output_video_writer, save_video_flag
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])

    if not video_path:
        return  # If no file is selected, return early.

    # Release the previous video if any
    if video_cap is not None:
        video_cap.release()

    # Open the new video
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        messagebox.showerror("Error", "Unable to open video file")
        return

    # If save video flag is enabled, setup video writer for output
    if save_video_flag:
        # Set up the output video writer for saving processed video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec
        output_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi")])
        if output_path:
            frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    print(f"Video file loaded: {video_path}")

    # Start video playback
    play_video()


def download_youtube_video():
    global video_cap
    # Prompt the user for a YouTube URL
    youtube_url = simpledialog.askstring("YouTube URL", "Enter YouTube Video URL:")
    if youtube_url:
        # Download the YouTube video with the best format that includes both video and audio
        ydl_opts = {
            'format': 'best',  # Download the best available format with both video and audio
            'outtmpl': 'downloaded_video.%(ext)s',
            'noplaylist': True
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(youtube_url, download=True)
                video_filename = ydl.prepare_filename(info_dict)

            # Load the downloaded video
            video_cap = cv2.VideoCapture(video_filename)
            if not video_cap.isOpened():
                messagebox.showerror("Error", "Unable to open downloaded YouTube video")
                return

            play_video()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to download video: {e}")


def open_camera():
    def select_camera():
        global video_cap
        selected_camera = selected_cam.get()  # Get the selected camera index
        # Release any previously opened video capture
        if video_cap is not None:
            video_cap.release()
        video_cap = cv2.VideoCapture(selected_camera)
        if not video_cap.isOpened():
            messagebox.showerror("Error", "Unable to open selected camera")
        else:
            camera_window.destroy()
            play_video()

    # Scan for attached cameras (typically numbered 0, 1, 2, etc.)
    cameras = []
    for i in range(10):  # Check for 10 cameras (adjust if you expect more cameras)
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cameras.append(i)
        cap.release()

    if not cameras:
        messagebox.showerror("No Cameras", "No cameras found on the device")
        return

    # Create a new window for camera selection
    camera_window = Toplevel(root)
    camera_window.title("Select Camera")
    camera_window.grab_set()  # Ensure the popup is in focus and blocks interaction with the main window
    camera_window.focus_force()  # Focus on the new window
    camera_window.attributes("-topmost", True)  # Ensure the window stays on top
    selected_cam = IntVar()  # Variable to store selected camera

    for cam_index in cameras:
        tk.Radiobutton(camera_window, text=f"Camera {cam_index}", variable=selected_cam, value=cam_index).pack(anchor=tk.W)

    tk.Button(camera_window, text="Select", command=select_camera).pack()





def pause_video():
    global playing
    playing = False


def toggle_save_video():
    global save_video_flag
    save_video_flag = not save_video_flag  # Toggle the flag

def init_gui():
    global root, screen_width, screen_height, control_panel, display_frame, label_count_display
    global threshold_slider, nms_slider  # Declare them global here

    # Initialize the main window
    root = tk.Tk()
    root.title("Video Segmentation with YOLO")

    # Get the screen resolution
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Apply the resolution to the main window (make it non-resizable)
    root.geometry(f"{screen_width}x{screen_height}")
    root.resizable(False, False)  # Disable resizing at the main window level
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=10)  # Control panel 15%
    root.grid_columnconfigure(1, weight=80)  # Main video display 75%
    root.grid_columnconfigure(2, weight=10)  # Report panel 15%

    # Create control panel, sliders, and buttons
    control_panel = tk.Frame(root, bg="lightgray", width=200, height=screen_height)
    control_panel.grid(row=0, column=0, sticky="nswe")
    control_panel.grid_propagate(False)

    # First Slider (NMS Threshold)
    nms_label = tk.Label(control_panel, text="NMS Threshold", bg="lightgray", fg="blue", font=("Helvetica", 12))  # Blue text
    nms_label.pack(pady=5)
    nms_slider = tk.Scale(control_panel, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, bg="white", fg="blue", length=180)  # Blue scale
    nms_slider.pack(padx=10)

    # Second Slider (Confidence Threshold)
    threshold_label = tk.Label(control_panel, text="Threshold", bg="lightgray", fg="blue", font=("Helvetica", 12))  # Blue text
    threshold_label.pack(pady=5)
    threshold_slider = tk.Scale(control_panel, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, bg="white", fg="blue", length=180)  # Blue scale
    threshold_slider.pack(padx=10)

    # Save Video Checkbox
    save_video_checkbox = tk.Checkbutton(control_panel, text="Save Video", bg="lightgray", fg="black", font=("Helvetica", 12), command=toggle_save_video)
    save_video_checkbox.pack(pady=10)

    # Add video control buttons with colored backgrounds
    btn_open_video = tk.Button(control_panel, text="Open Video", bg="blue", fg="white", font=("Helvetica", 12), height=2, width=15, command=open_video_file)
    btn_open_video.pack(pady=10)

    btn_youtube_video = tk.Button(control_panel, text="YouTube Videos", bg="purple", fg="white", font=("Helvetica", 12), height=2, width=15, command=download_youtube_video)
    btn_youtube_video.pack(pady=10)

    btn_open_camera = tk.Button(control_panel, text="Open Camera", bg="yellow", fg="black", font=("Helvetica", 12), height=2, width=15, command=open_camera)
    btn_open_camera.pack(pady=10)

    btn_play_video = tk.Button(control_panel, text="Play", bg="green", fg="white", font=("Helvetica", 12), height=2, width=15, command=play_video)
    btn_play_video.pack(pady=10)

    btn_pause_video = tk.Button(control_panel, text="Pause", bg="red", fg="white", font=("Helvetica", 12), height=2, width=15, command=pause_video)
    btn_pause_video.pack(pady=10)

    # Create display frame for video
    display_frame = tk.Frame(root, bg="white")
    display_frame.grid(row=0, column=1, sticky="nsew")
    display_frame.grid_rowconfigure(0, weight=1)
    display_frame.grid_columnconfigure(0, weight=1)
    panel = tk.Label(display_frame, bg="white")
    panel.grid(row=0, column=0, sticky="nsew")

    # Create label frame for report panel (15%)
    label_frame = tk.Frame(root, bg="lightgray")
    label_frame.grid(row=0, column=2, sticky="nswe")
    label_frame.grid_propagate(False)
    label_count_display = tk.Label(label_frame, text="", bg="lightgray", font=("Helvetica", 10), anchor='nw', justify='left')
    label_count_display.pack(padx=10, pady=10, fill="both", expand=True)

    # Run the Tkinter loop
    root.mainloop()


def play_video():
    global playing, video_cap, current_frame, threshold_slider, nms_slider  # Declare them as global
    playing = True
    if video_cap is not None:
        while playing and video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                print("End of video or error reading the frame")
                playing = False
                break

            current_frame = frame  # Store the current frame for refresh
            print("Processing frame...")

            # Apply YOLO predictions on the frame
            threshold_value = threshold_slider.get()
            nms_value = nms_slider.get()
            processed_frame = perform_yolo_prediction(frame, display_frame, panel, label_counts_dict, label_count_display, threshold_value, nms_value)

            # Write the processed frame to the output video if saving is enabled
            if save_video_flag and output_video_writer is not None:
                output_video_writer.write(processed_frame)

            # Update the GUI and wait for the next frame
            root.update()
            cv2.waitKey(int(1000 / frame_rate))  # Adjust frame rate


if __name__ == "__main__":
    multiprocessing.freeze_support()
    init_gui()

