from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os

app = FastAPI()

# --- CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = os.path.join("..", "Model", "asl_model.h5")
LABEL_PATH = os.path.join("..", "Model", "labels.pickle")

# --- LOAD RESOURCES ---
model = None
labels_dict = {}

try:
    # 1. Load the Model
    # compile=False avoids potential errors with custom optimizers in older TF versions
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # 2. Load the Pickle Labels
    with open(LABEL_PATH, 'rb') as f:
        labels_dict = pickle.load(f)
        
    print("✅ Custom ASL Model & Labels loaded successfully!")
    
except Exception as e:
    print(f"❌ Critical Error loading model/labels: {e}")

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
# static_image_mode=True is BETTER for single image uploads (like we use in the backend)
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

@app.get("/")
def home():
    return {"message": "Custom ASL Backend is Ready"}

@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # 1. Read Image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2. Prepare for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Detect Hand
    results = hands.process(img_rgb)
    
    # If no hand found, return early
    if not results.multi_hand_landmarks:
        return {
            "prediction": "No Hand Detected", 
            "confidence": 0.0,
            "index": -1
        }

    # 4. Extract Landmarks (The exact logic from your run_model.py)
    data_aux = []
    
    # We only take the first hand detected
    hand_landmarks = results.multi_hand_landmarks[0]
    
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x)
        data_aux.append(y)
    
    # 5. Predict
    # Model expects shape (1, 42) -> 21 points * 2 coords (x, y)
    try:
        prediction = model.predict(np.array([data_aux]), verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Get the character from your dictionary
        predicted_char = labels_dict[predicted_index]

        return {
            "prediction": predicted_char,
            "confidence": confidence,
            "index": int(predicted_index)
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {
            "prediction": "Error", 
            "confidence": 0.0
        }