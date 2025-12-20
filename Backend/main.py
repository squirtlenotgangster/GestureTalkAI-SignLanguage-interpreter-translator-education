from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os

# Import our new database files
import models, schemas, database

# --- DATABASE SETUP ---
models.Base.metadata.create_all(bind=database.engine)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- SECURITY SETUP ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# --- APP SETUP ---
app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI SETUP ---
MODEL_PATH = os.path.join("..", "Model", "asl_model.h5")
LABEL_PATH = os.path.join("..", "Model", "labels.pickle")

model = None
labels_dict = {}

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABEL_PATH, 'rb') as f:
        labels_dict = pickle.load(f)
    print("✅ Custom ASL Model & Labels loaded successfully!")
except Exception as e:
    print(f"❌ Critical Error loading model: {e}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


# --- ROUTES ---

@app.get("/")
def home():
    return {"message": "Backend is Running"}

# 1. REGISTER
@app.post("/register", response_model=schemas.UserResponse)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if username exists
    if db.query(models.User).filter(models.User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Check if email exists
    if db.query(models.User).filter(models.User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = get_password_hash(user.password)
    new_user = models.User(username=user.username, email=user.email, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# 2. LOGIN
@app.post("/login")
def login_user(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    return {"message": "Login successful", "username": db_user.username}

# 3. CONTACT
@app.post("/contact")
def submit_contact(contact: schemas.ContactCreate, db: Session = Depends(get_db)):
    new_msg = models.ContactMessage(name=contact.name, email=contact.email, message=contact.message)
    db.add(new_msg)
    db.commit()
    return {"message": "Message sent successfully"}

# 4. PREDICT
@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return {"prediction": "No Hand Detected", "confidence": 0.0}

    data_aux = []
    hand_landmarks = results.multi_hand_landmarks[0]
    for i in range(len(hand_landmarks.landmark)):
        data_aux.append(hand_landmarks.landmark[i].x)
        data_aux.append(hand_landmarks.landmark[i].y)
    
    try:
        prediction = model.predict(np.array([data_aux]), verbose=0)
        index = np.argmax(prediction)
        return {"prediction": labels_dict[index], "confidence": float(np.max(prediction))}
    except Exception:
        return {"prediction": "Error", "confidence": 0.0}