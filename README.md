# 🌱 AgroGuard

**AgroGuard** is an AI-powered web application that detects crop diseases from leaf images using Deep Learning and Computer Vision. Built with **Flask** and **TensorFlow/Keras**, the application allows users to upload a crop leaf image, automatically predicts the disease, and provides an easy-to-use interface for rapid diagnosis.

The project aims to support farmers and agricultural researchers by enabling early disease detection, helping reduce crop losses and improve productivity through timely intervention.
---
## ✨ Features

- 🌿 Upload crop leaf images through a simple web interface.
- 🤖 AI-powered crop disease prediction using a trained deep learning model.
- ⚡ Fast image processing and prediction.
- 📊 Displays the predicted crop disease.
- 💻 Easy-to-use Flask web application.
- 📱 Clean and responsive user interface.
---

## 🛠️ Tech Stack

### Programming Language
- Python

### Backend Framework
- Flask

### Deep Learning
- TensorFlow
- Keras

### Image Processing
- OpenCV
- Pillow
- NumPy

### Frontend
- HTML
- CSS
- JavaScript
---

## 🔄 Project Workflow

1. User uploads a crop leaf image.
2. Flask receives the uploaded image.
3. The image is preprocessed and resized.
4. The trained deep learning model analyzes the image.
5. The predicted crop disease is generated.
6. The prediction result is displayed to the user.
---

## 📂 Project Structure

```text
AgroGuard/
│
├── model/                 # Trained deep learning model
├── static/                # CSS, JavaScript, images, uploads
├── templates/             # HTML templates
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── class_names.txt        # Disease class labels
├── requirements.txt       # Project dependencies
├── runtime.txt            # Python runtime version
├── Procfile               # Deployment configuration
└── README.md              # Project documentation
```
