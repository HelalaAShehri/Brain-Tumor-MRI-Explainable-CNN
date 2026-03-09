# Brain Tumor MRI Classification with Explainable CNN

This repository provides the **pretrained CNN model and demonstration notebook** accompanying the research study:

**“An Interpretable Lightweight CNN Framework for Multi-Class Brain Tumor Classification from MRI.”**

The repository allows users to reproduce the **inference and explainability results** using Grad-CAM.

---

## Repository Structure

- **Desktop_App/**
  - Contains a lightweight desktop application for running the model locally.
  - Includes the main interface script (`MudrekApp.py`) along with supporting assets such as fonts, icons, and interface resources.

- **colab/**
  - Contains the Google Colab notebook used for inference and explainability.
  - **Brain_Tumor_MRI_GradCAM_Demo.ipynb** demonstrates how to:
    - download the pretrained model,
    - perform brain tumor classification,
    - visualize Grad-CAM explanations.

- **figures/**
  - Stores figures used in the repository documentation.
  - **gradcam_example.png** shows an example Grad-CAM visualization highlighting tumor regions in MRI images.

- **sample_images/**
  - Example MRI images used for demonstration and testing.
  - Includes representative samples from each class:
    - glioma
    - meningioma
    - pituitary
    - no tumor

- **requirements.txt**
  - Lists the Python dependencies required to run the inference notebook.

- **README.md**
  - Provides an overview of the project, usage instructions, and example visualizations.

---

## Model Weights

Due to GitHub file size limitations, the pretrained model is hosted on Google Drive:

🔗 [Download pretrained model](https://drive.google.com/file/d/1B9bfeIz6B3rUxW2LJg7Ze850_wde8ade/view)

The Google Colab notebook **automatically downloads** the model before running inference.

---

## Model Input Specifications

| Property | Value |
|----------|-------|
| Image resolution | 125 × 125 pixels |
| Channels | 1 (grayscale) |
| Classes | glioma, meningioma, pituitary, notumor |

---

## Running the Demo

1. Open the notebook: [`colab/Brain_Tumor_MRI_GradCAM_Demo.ipynb`](colab/Brain_Tumor_MRI_GradCAM_Demo.ipynb)
2. Run the model download cell
3. Load the pretrained model
4. Upload or select MRI images from `sample_images/`
5. Run inference
6. Visualize Grad-CAM attention maps

---

## Model Input

The model expects **grayscale MRI images** with:
Resolution: 125 × 125
Channels: 1


Classes: glioma, meningioma, pituitary, notumor

---

## Running the Demo

Open the notebook: colab/Brain_Tumor_MRI_GradCAM_Demo.ipynb

Steps:

1. Run the model download cell
2. Load the pretrained model
3. Upload or select MRI images
4. Run inference
5. Visualize Grad-CAM attention maps

---

## Explainability

Grad-CAM is used to visualize the regions of the MRI contributing to the model prediction.

To improve interpretability, Grad-CAM heatmaps are constrained using a brain mask, ensuring attention is localized within the anatomical brain region.

### Example Visualization

![GradCAM Example](figures/gradcam_example.png)

---

## Disclaimer

This repository is intended **for research purposes only** and should not be used for clinical diagnosis. The model and code are not approved for clinical use and should not be used for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.


