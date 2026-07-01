# 🧠 Animal Image Classification using Convolutional Neural Network (CNN)

## 📌 Overview

This project is an **Animal Image Classification System** developed using a **Convolutional Neural Network (CNN)**. The application allows users to upload an animal image and automatically predicts the animal class through a user-friendly **Streamlit** web interface.

The project demonstrates the implementation of Deep Learning for image classification, including image preprocessing, CNN model training, and deployment using Streamlit.

---

## 📷 Application Preview

### 🏠 Home Page

![Home](screenshots/home.png)

### 📤 Upload Image

![Upload](screenshots/upload.png)

### 📊 Prediction Result

![Prediction](screenshots/prediction.png)

---

## ✨ Features

- Upload animal images (.jpg, .png)
- Image classification using CNN
- Predict one of 10 animal classes
- Display prediction confidence
- Interactive web interface built with Streamlit
- Fast and easy-to-use prediction system

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Pillow

---

## 📂 Project Structure

```text
animal-image-classification-cnn/
│
├── screenshots/
│   ├── home.png
│   ├── upload.png
│   └── prediction.png
│
├── app.py
├── classes.npy
├── klasifikasi_hewan.ipynb
├── model_hewan.h5
├── requirements.txt
└── README.md
```

---

## 🐾 Animal Classes

The model is trained to classify the following animal classes:

- Butterfly
- Cat
- Chicken
- Cow
- Dog
- Elephant
- Horse
- Sheep
- Spider
- Squirrel

---

## 📊 Model Information

| Item | Description |
|------|-------------|
| Model | Convolutional Neural Network (CNN) |
| Number of Classes | 10 |
| Input Image Size | 150 × 150 pixels |
| Framework | TensorFlow / Keras |
| Interface | Streamlit |

---

## 📁 Dataset

The dataset used in this project was obtained from **Kaggle** and was used for educational and portfolio purposes.

---

## 🚀 Installation

### Clone this repository

```bash
git clone https://github.com/deviagung01/animal-image-classification-cnn.git
```

### Move into the project directory

```bash
cd animal-image-classification-cnn
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 🎯 Project Objectives

- Learn Deep Learning implementation using CNN.
- Build an image classification application.
- Apply TensorFlow and Streamlit in a real project.
- Develop an interactive AI-based web application.

---

## 👨‍💻 Author

**Devi Agung Kristanto**

Information Technology Student  
Machine Learning & Deep Learning Enthusiast

GitHub:
https://github.com/deviagung01

---

## 📄 License

This project is developed for educational, research, and portfolio purposes.
