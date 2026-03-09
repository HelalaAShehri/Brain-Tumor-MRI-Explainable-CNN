# Import necessary libraries for GUI, deep learning, image processing, and file handling
import tkinter as tk  # Base Tkinter module for GUI
from typing import Callable  # For type hinting function callbacks
import customtkinter as ctk  # CustomTkinter for enhanced themed widgets
import numpy as np  # Numerical computing
from tkinter import *  # Import all Tkinter components
from tkinter import messagebox, filedialog  # To show messages and allow file uploads
from customtkinter import *  # Import all CustomTkinter components
from PIL import *  # PIL for image processing
from PIL import Image, ImageTk  # Specifically for image display and conversion in Tkinter
from tkvideo import tkvideo  # For playing video in the GUI
from tensorflow.keras.models import load_model  # Load saved deep learning models
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Load and convert images
import os  # OS utilities for path management
import sys  # System functions
import cv2  # OpenCV for image processing
from tensorflow.keras import backend as K  # Backend operations
import tensorflow as tf  # TensorFlow deep learning library

# Generate Grad-CAM heatmap for visual explanation
def generate_gradcam(model, img_array, layer_name):
    # Create a model that outputs both the target layer and final prediction
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])  # Identify the class index with highest probability
        loss = predictions[:, class_idx]  # Focus on the relevant class prediction
    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Average gradients over feature map
    conv_outputs = conv_outputs[0]
    # Generate weighted combination of feature maps
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)  # Apply ReLU
    heatmap /= (tf.math.reduce_max(heatmap) + K.epsilon())  # Normalize
    return heatmap.numpy()  # Return as NumPy array

# Overlay Grad-CAM heatmap onto the original image
def overlay_heatmap(img_path, heatmap, alpha=0.4, output_size=(250, 250)):
    original = cv2.imread(img_path)  # Load original image
    original = cv2.resize(original, output_size)  # Resize image
    heatmap_resized = cv2.resize(heatmap, output_size)  # Resize heatmap to match
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)  # Apply color map
    overlayed = cv2.addWeighted(original, 1 - alpha, heatmap_color, alpha, 0)  # Merge images
    return Image.fromarray(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))  # Convert to RGB for display

# Set the base directory, accounting for PyInstaller environments
base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# Load pre-trained models
ad_model_path = os.path.join(base_dir, 'SavedFinalADModel.keras')
tumor_model_path = os.path.join(base_dir, 'FinalModelBrain.keras')
ad_model = load_model(ad_model_path)
tumor_model = load_model(tumor_model_path)

# Define label classes
ad_labels = ['CN','EMCI','LMCI','MCI','AD']  # Alzheimer disease stages
tumor_labels = ['notumor', 'pituitary', 'meningioma', 'glioma']  # Brain tumor classes

# Initialize global variables for UI elements and logic
uploaded_image_path = None
uploaded_image_lbl= None
result_label=None
uploaded_image=None
prediction_label = None
ad_predicted_label=None
model=None
target_size=None
text_color=None
new_ad_predicted_label=None
new_tumor_predicted_label=None

# Create the main application window
app = ctk.CTk()
app.title("Mudrek - Brain Disease Classification")  # Set title
app.iconbitmap("logo.ico")  # Set window icon
logo_image = ctk.CTkImage(Image.open("logo.png"), size=(60, 45))  # Load logo for navigation

# Set app size to full screen
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
app.geometry(f"{screen_width}x{screen_height}")

# Utility to create custom fonts
def custom_font(size:int):
    return ctk.CTkFont(family="MadaniArabic-Medium", size=size)

# Creates a standardized button
def get_button(text:str, command:Callable):
    return CTkButton(
        master=app, text=text, font=custom_font(16), corner_radius=32,
        bg_color="black", fg_color="#0A325E", hover_color="#3199D4",
        height=50, width=160, command=command
    )

# Sets upload and scan buttons with proper placement
def set_buttons(scan_comand:Callable):
    upload_button = get_button(text="Upload Image", command=upload_image)
    upload_button.place(relx=0.4, rely=0.85, anchor="center")

    scan_button = get_button(text="Classify Image", command=scan_comand)
    scan_button.place(relx=0.6, rely=0.85, anchor="center")

# Destroys all UI components in the window (used for switching pages)
def destroy_all_widgets():
    for widget in app.winfo_children():
        widget.destroy()

# Navigation bar with hover effects and logo/home button
def nav_bar():
    global screen_width

    # Hover styling
    def info_on_enter(event): info_button.configure(text_color='#3199D4')
    def info_on_leave(event): info_button.configure(text_color='white')
    def about_on_enter(event): about_us_button.configure(text_color='#3199D4')
    def about_on_leave(event): about_us_button.configure(text_color='white')

    nav_frame = ctk.CTkFrame(app, bg_color="black", fg_color="black", width=screen_width, height=80)
    nav_frame.place(x=0, y=0, relwidth=1)

    about_us_button = ctk.CTkButton(nav_frame, text="About Us", font=custom_font(16),
                                    fg_color="transparent", command=about_us)
    about_us_button.bind('<Enter>', about_on_enter, add='+')
    about_us_button.bind("<Leave>", about_on_leave, add='+')
    about_us_button.pack(side=tk.LEFT, padx=10 , pady=20)

    info_button = ctk.CTkButton(nav_frame, text="Information", font=custom_font(16),
                                fg_color="transparent", command=information)
    info_button.bind('<Enter>', info_on_enter, add='+')
    info_button.bind("<Leave>", info_on_leave, add='+')
    info_button.pack(side=tk.LEFT, padx=10)

    # Logo/home button
    home_button = ctk.CTkButton(nav_frame, text="", image=logo_image,
                                fg_color="transparent", hover=False, command=main_screen)
    home_button.pack(side=tk.RIGHT, padx=0)
# Function to handle image upload from local files
def upload_image():
    global uploaded_image_path, uploaded_image_lbl, result_label, uploaded_image, prediction_label

    # Open file dialog for image selection
    uploaded_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg; *.png")])

    # Validate file format
    if not uploaded_image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        messagebox.showwarning("Unsupported Format", "Please upload a valid image file (.png, .jpg, .jpeg).")
        return

    if uploaded_image_path:
        # Resize and prepare image for display
        uploaded_image = ImageTk.PhotoImage(Image.open(uploaded_image_path).resize((250, 250)))

        # Update existing label with new image
        if uploaded_image_lbl is not None:
            uploaded_image_lbl.configure(image=uploaded_image)
            uploaded_image_lbl.image = uploaded_image  # Prevent garbage collection

        # Reset result label if already created
        if result_label is not None:
            result_label.destroy()
            result_label = None

        # Create and place result message label
        if result_label is None:
            result_label = ctk.CTkLabel(app, text="Image uploaded successfully! click on scan.",
                                        font=custom_font(15), text_color="white", bg_color="black")
            result_label.place(relx=0.5, rely=0.70, anchor="center")
        result_label.configure(text="Image uploaded successfully! click on scan.",
                               font=custom_font(15), text_color="white")

# Upload & classify image for tumor detection
def upload_tumor():
    global uploaded_image_path, uploaded_image_lbl, result_label, uploaded_image

    destroy_all_widgets()  # Clear existing widgets
    get_frame_with_label("Mudrek Interfaces/2.png")  # Load background image
    nav_bar()  # Load navigation bar
    set_buttons(scan_comand=scan_tumor)  # Set buttons for scanning

    # Placeholder for image preview
    uploaded_image_lbl = ctk.CTkLabel(app, text="", width=300, height=300, bg_color="black")
    uploaded_image_lbl.place(relx=0.5, rely=0.47, anchor="center")

    # Prediction label placeholder
    prediction_label = ctk.CTkLabel(app, text="", font=custom_font(15), text_color="white", bg_color="black")
    prediction_label.place(relx=0.5, rely=0.64, anchor="center")

    upload_image()  # Trigger image upload

# Upload & classify image for Alzheimer's disease detection
def upload_ad():
    global uploaded_image_path, uploaded_image_lbl, result_label, new_ad_predicted_label, prediction_text_color

    destroy_all_widgets()
    get_frame_with_label("Mudrek Interfaces/4.png")
    nav_bar()
    set_buttons(scan_comand=scan_ad)

    uploaded_image_lbl = ctk.CTkLabel(app, text="", width=300, height=300, bg_color="black")
    uploaded_image_lbl.place(relx=0.5, rely=0.47, anchor="center")

    prediction_label = ctk.CTkLabel(app, text="", font=custom_font(15), text_color="white", bg_color="black")
    prediction_label.place(relx=0.5, rely=0.64, anchor="center")

    upload_image()

# Set a background image as a frame for GUI sections
def get_frame_with_label(background_path: str):
    frame = ctk.CTkFrame(app).pack(fill="both", expand=True)  # Expand to fill window
    bg_image = ImageTk.PhotoImage(Image.open(background_path))  # Load background
    bg_lbl = ctk.CTkLabel(frame, text="", image=bg_image)
    bg_lbl.place(relwidth=1, relheight=1)  # Fill the frame completely

# Alzheimer image scan prediction logic
def scan_ad():
    global uploaded_image_path, uploaded_image_lbl, result_label, uploaded_image, prediction_label, new_ad_predicted_label

    if not uploaded_image_path:
        messagebox.showwarning("Warning", "Please upload an image first.")
        return

    # Reset labels
    result_label.configure(text="") if result_label else None
    result_label = None
    prediction_label = None

    target_size = (150, 150)  # Resize image
    try:
        # Load and preprocess image
        image = load_img(uploaded_image_path, target_size=target_size, color_mode='grayscale')
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
    except Exception as e:
        result_label.configure(text=f"Error loading image: {e}")

    # Predict using AD model
    predictions = ad_model.predict(image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    ad_predicted_label = ad_labels[predicted_class_index]

    # Map prediction to readable label and color
    if ad_predicted_label == 'CN':
        new_ad_predicted_label = "Cognitively Normal"
        prediction_text_color = "#00a000"
    elif ad_predicted_label == 'EMCI':
        new_ad_predicted_label = " Early Mild Cognitive Impairment "
        prediction_text_color = "#80c300"
    elif ad_predicted_label == 'MCI':
        new_ad_predicted_label = " Mild Cognitive Impairment "
        prediction_text_color = "#ffe500"
    elif ad_predicted_label == 'LMCI':
        new_ad_predicted_label = "Late Mild Cognitive Impairment "
        prediction_text_color = "#ff9300"
    elif ad_predicted_label == 'AD':
        new_ad_predicted_label = " Alzheimer's Disease "
        prediction_text_color = "#ff0000"

    # Display predicted diagnosis
    if prediction_label is None:
        prediction_label = ctk.CTkLabel(app, text="Predicted Diagnosis:", font=custom_font(15),
                                        text_color="white", bg_color="black")
        prediction_label.place(relx=0.5, rely=0.64, anchor="center")
    prediction_label.configure(text="Predicted Diagnosis:", text_color="white")

    if result_label is None:
        result_label = ctk.CTkLabel(app, text=new_ad_predicted_label, font=custom_font(15),
                                    text_color=prediction_text_color, bg_color="black")
        result_label.place(relx=0.5, rely=0.68, anchor="center")
    result_label.configure(text=new_ad_predicted_label, text_color=prediction_text_color)
# Function to perform brain tumor classification and Grad-CAM visualization
def scan_tumor():
    global uploaded_image_path, uploaded_image_lbl, result_label, prediction_label, new_tumor_predicted_label, confidence_label

    if not uploaded_image_path:
        messagebox.showwarning("Warning", "Please upload an image first.")
        return

    # Clear previous labels if they exist
    if result_label: result_label.destroy()
    if prediction_label: prediction_label.destroy()
    if 'confidence_label' in globals() and confidence_label: confidence_label.destroy()

    # Preprocess the input image
    target_size = (125, 125)
    image = load_img(uploaded_image_path, target_size=target_size, color_mode='grayscale')
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction using the tumor model
    predictions = tumor_model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    tumor_predicted_label = tumor_labels[predicted_class_index]

    # Calculate confidence level
    confidence = float(np.max(predictions)) * 100
    confidence_text = f"Confidence: {confidence:.2f}%"

    # Label text and associated colors
    label_color_map = {
        'notumor': ("No Tumor", "#00a000"),
        'pituitary': ("Pituitary", "#ffe500"),
        'meningioma': ("Meningioma", "#ff9300"),
        'glioma': ("Glioma", "#ff0000")
    }
    new_tumor_predicted_label, prediction_text_color = label_color_map[tumor_predicted_label]

    # Grad-CAM visualization
    try:
        gradcam_layer = 'conv2d_13'  # Final convolutional layer name
        heatmap = generate_gradcam(tumor_model, image_array, gradcam_layer)
        overlay_image = overlay_heatmap(uploaded_image_path, heatmap, output_size=(250, 250))
        gradcam_tk_image = ImageTk.PhotoImage(overlay_image)

        if uploaded_image_lbl:
            uploaded_image_lbl.configure(image=gradcam_tk_image)
            uploaded_image_lbl.image = gradcam_tk_image  # Preserve reference

    except Exception as e:
        messagebox.showerror("Grad-CAM Error", f"Failed to generate Grad-CAM: {e}")

    # Display confidence level
    confidence_label = ctk.CTkLabel(app, text=confidence_text, font=custom_font(12), text_color="white", bg_color="black")
    confidence_label.place(relx=0.5, rely=0.63, anchor="center")

    # Display "Predicted Diagnosis:" label
    prediction_label = ctk.CTkLabel(app, text="Predicted Diagnosis:", font=custom_font(15), text_color="white", bg_color="black")
    prediction_label.place(relx=0.5, rely=0.66, anchor="center")

    # Display the prediction result
    result_label = ctk.CTkLabel(app, text=new_tumor_predicted_label, font=custom_font(15), text_color=prediction_text_color, bg_color="black")
    result_label.place(relx=0.5, rely=0.70, anchor="center")

# Alzheimer button navigates to upload/scan AD page
def alzheimer_disease():
    destroy_all_widgets()
    get_frame_with_label("Mudrek Interfaces/3.png")
    nav_bar()

    upload_button = CTkButton(master=app, text="Upload Image", font=custom_font(16), corner_radius=32,
                               bg_color="black", fg_color="#0A325E", hover_color="#3199D4", height=50, width=160,
                               command=upload_ad)
    upload_button.place(relx=0.4, rely=0.85, anchor="center")

    scan_button = CTkButton(master=app, text="Classify Image", font=custom_font(16), corner_radius=32,
                             bg_color="black", fg_color="#0A325E", hover_color="#3199D4", height=50, width=160,
                             command=scan_ad)
    scan_button.place(relx=0.6, rely=0.85, anchor="center")

# Brain tumor button navigates to upload/scan tumor page
def brain_tumor():
    destroy_all_widgets()
    get_frame_with_label("Mudrek Interfaces/1.png")
    nav_bar()

    upload_button = CTkButton(master=app, text="Upload Image", font=custom_font(16), corner_radius=32,
                               bg_color="black", fg_color="#0A325E", hover_color="#3199D4", height=50, width=160,
                               command=upload_tumor)
    upload_button.place(relx=0.4, rely=0.85, anchor="center")

    scan_button = CTkButton(master=app, text="Classify Image", font=custom_font(16), corner_radius=32,
                             bg_color="black", fg_color="#0A325E", hover_color="#3199D4", height=50, width=160,
                             command=scan_tumor)
    scan_button.place(relx=0.6, rely=0.85, anchor="center")

# Display the Information page
def information():
    global screen_height, screen_width
    destroy_all_widgets()

    frame = ctk.CTkFrame(app)
    frame.pack(fill="both", expand=True)

    bg_image = ctk.CTkImage(Image.open("Mudrek Interfaces/6.png"), size=(screen_width, screen_height))
    lbl = ctk.CTkLabel(frame, text="", image=bg_image)
    lbl.pack(fill="both", expand=True)

    nav_bar()

# Display the About Us page
def about_us():
    global screen_height, screen_width
    destroy_all_widgets()

    frame = ctk.CTkFrame(app)
    frame.pack(fill="both", expand=True)

    bg_image = ctk.CTkImage(Image.open("Mudrek Interfaces/5.png"), size=(screen_width, screen_height))
    lbl = ctk.CTkLabel(frame, text="", image=bg_image)
    lbl.pack(fill="both", expand=True)

    nav_bar()

# Main screen setup with video background and homepage navigation
def main_screen():
    global screen_height, screen_width, uploaded_image_path
    uploaded_image_path = None
    destroy_all_widgets()

    frame = ctk.CTkFrame(app)
    frame.pack(fill="both", expand=True)
    lbl = tk.Label(frame)
    lbl.pack(fill=tk.BOTH, expand=True)

    def resize_video(event=None):
        lbl.configure(width=screen_width, height=screen_height)

    app.bind("<Configure>", resize_video)

    player = tkvideo("Mudrek Interfaces/0.mp4", lbl, loop=1, size=(screen_width+400, screen_height+200))
    player.play()

    nav_bar()

    # Main action buttons
    brain_tumor_button = CTkButton(master=app, text="Brain Tumor", font=custom_font(16), corner_radius=32,
                                    bg_color="black", fg_color="#0A325E", hover_color="#3199D4", height=50, width=160,
                                    command=brain_tumor)
    brain_tumor_button.place(relx=0.40, rely=0.8, anchor="center")

    alzheimer_button = CTkButton(master=app, text="Early Alzheimer", font=custom_font(16), corner_radius=32,
                                  bg_color="black", fg_color="#0A325E", hover_color="#3199D4", height=50, width=160,
                                  command=alzheimer_disease)
    alzheimer_button.place(relx=0.60, rely=0.8, anchor="center")

# Start the application with the main screen
main_screen()
app.mainloop()
