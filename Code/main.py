from fastapi import FastAPI,File, UploadFile,Form
import cv2 as cv2
import os
import numpy as np
from model_util import *


database={}
app = FastAPI()

@app.get('/')
def health_check():
    return {'health_check' : 'OK'}

@app.get('/info')
def info():
    return {'name': 'Face_Detect_Verify', 'description': "A model using One-Shot Learning to Identify and Verify Indviduals" }


@app.post("/detect_verify")
async def verify_face(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image"}
    faces_cord = detect_verify(frame, database) 
    return {"status": "success", "faces": faces_cord}

@app.post("/upload_image")
async def upload_image(name: str = Form(...), image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode image into OpenCV format
    if frame is None:
        return {"error": "Invalid image file"}
    encodded=encode(frame)
    database[name]=encodded #np.array(encodded['output_0'])
    return {"status": "success", "message": f"{name} added to database"}
