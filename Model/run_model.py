import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = './asl_model.h5'
LABEL_PATH = './labels.pickle'
MIN_CONFIDENCE = 0.5  # Threshold to detect hand

# --- WORD BUILDER VARIABLES ---
current_word = ""
sentence = ""
frame_counter = 0
last_prediction = None
STABILITY_THRESHOLD = 15  # Frames to hold sign before registering (approx 0.5 sec)

# 1. Load the Model and Labels
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    labels_dict = pickle.load(f)

print("Model loaded successfully!")

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# static_image_mode=False is CRITICAL for webcam (it tracks movement faster)
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1, 
                       min_detection_confidence=MIN_CONFIDENCE)

# 3. Start Webcam
cap = cv2.VideoCapture(0) # 0 is usually the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame for mirror effect (natural feel)
    frame = cv2.flip(frame, 1)
    
    # Get frame dimensions
    H, W, _ = frame.shape
    
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands.process(frame_rgb)
    
    # If a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # --- Draw the Skeleton ---
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # --- Prepare Data for Model ---
            data_aux = []
            x_vals = []
            y_vals = []
            
            # Extract Coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_vals.append(x)
                y_vals.append(y)
            
            # --- Bounding Box Logic ---
            x1 = int(min(x_vals) * W) - 10
            y1 = int(min(y_vals) * H) - 10
            x2 = int(max(x_vals) * W) + 10
            y2 = int(max(y_vals) * H) + 10
            
            # --- Prediction ---
            prediction = model.predict(np.array([data_aux]), verbose=0)
            predicted_index = np.argmax(prediction)
            predicted_character = labels_dict[predicted_index]
            confidence = np.max(prediction)
            
            # --- STABILITY & WORD BUILDING LOGIC ---
            # Only count frames if the prediction matches the previous frame
            if predicted_character == last_prediction:
                frame_counter += 1
            else:
                frame_counter = 0
                last_prediction = predicted_character
            
            # If threshold reached, register the character
            if frame_counter == STABILITY_THRESHOLD:
                
                # 'space' -> Adds space to sentence, resets current word
                if predicted_character == 'space':
                    sentence += current_word + " "
                    current_word = "" # Clear current word after space
                
                # 'del' -> Removes last character
                elif predicted_character == 'del':
                    current_word = current_word[:-1]
                
                # 'nothing' -> Do nothing (idle state)
                elif predicted_character == 'nothing':
                    pass
                
                # Regular letters -> Add to current word
                else:
                    current_word += predicted_character

            # --- Visual Feedback on Box ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) # Black Border
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            
    # --- UI DISPLAY (Always draw this, even if no hand detected) ---
    
    # Top Banner for Full Sentence
    cv2.rectangle(frame, (0, 0), (W, 80), (245, 117, 16), -1) # Orange banner
    cv2.putText(frame, f"Sentence: {sentence}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Bottom Banner for Current Word being typed
    cv2.rectangle(frame, (0, H-60), (W, H), (0, 0, 0), -1) # Black banner
    cv2.putText(frame, f"Word: {current_word}", (20, H-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow('ASL Interpreter', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()