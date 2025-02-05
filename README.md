# Face Detection & Verification - FastAPI

This repository contains a **Face Detection & Verification** system built using **FastAPI**. The model utilizes **One-Shot Learning** with a deep learning-based **FaceNet** model to identify and verify individuals.

## Features
- Face detection using **OpenCV**
- Face embedding extraction using **FaceNet**
- One-shot learning for face verification
- FastAPI-based REST API for easy interaction

---

## Installation

### **1. Clone the Repository**
```sh
git clone https://github.com/Moaz0009/Face-Detection-Verification-FASTAPI.git
cd Face-Detection-Verification-FASTAPI
```

### **2. Install Dependencies**
Make sure you have Python installed, then run:
```sh
pip install -r requirements.txt
```

### **3. Run the FastAPI Server**
```sh
uvicorn main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

---

## API Endpoints

### **1. Health Check**
**Endpoint:** `GET /`

**Response:**
```json
{"health_check": "OK"}
```

### **2. Get API Information**
**Endpoint:** `GET /info`

**Response:**
```json
{"name": "Face_Detect_Verify", "description": "A model using One-Shot Learning to Identify and Verify Individuals"}
```

### **3. Upload a Face Image**
**Endpoint:** `POST /upload_image`

**Request:**
- `name` (form data) - Name of the person
- `image` (file) - Image file to upload

**Response:**
```json
{"status": "success", "message": "John Doe added to database"}
```

### **4. Detect & Verify Face**
**Endpoint:** `POST /detect_verify`

**Request:**
- `image` (file) - Image file to verify

**Response:**
```json
{"status": "success", "faces": {"John Doe": [x, y, w, h]}}
```

---

## How It Works
1. **Face Embeddings Generation:**
   - When an image is uploaded, the model extracts **face embeddings** using **FaceNet** and stores them in a database.
2. **Face Verification:**
   - When an image is sent for verification, the model extracts embeddings and compares them with stored embeddings using **Euclidean distance**.
3. **Face Detection:**
   - Uses OpenCV’s Haar cascade classifier to detect faces before encoding.

---

## Dependencies
- `FastAPI`
- `OpenCV`
- `TensorFlow/Keras`
- `NumPy`

Install them using:
```sh
pip install fastapi opencv-python-headless tensorflow numpy uvicorn
```

---

## Future Improvements
- Improve face recognition accuracy using different distance metrics.
- Add support for multiple faces in a single image.
- Implement database persistence instead of in-memory storage.

---

## Author
**Moaz Abdeljalil**

Feel free to contribute or report issues! 🚀

