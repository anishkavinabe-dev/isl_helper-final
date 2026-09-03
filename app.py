import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib

# Page config
st.set_page_config(page_title="ISL LIVE Translator", layout="centered")
st.title("🤟 ISL LIVE Translator - Fixed Analysis")
st.write("Show your ISL sign in camera - Auto normalized prediction")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('isl_model.pkl')

model = load_model()
st.info(f"Model trained on signs: {list(model.classes_)}")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Camera input - 100% stable, no WebRTC issues
img_file = st.camera_input("Show your ISL sign")

if img_file:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_coords = []
            y_coords = []

            # Step 1: Collect all x, y coordinates for normalization
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)

            # Step 2: Create normalized features (This is the main fix)
            # This must match your training logic
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_coords))
                    data_aux.append(lm.y - min(y_coords))
                    data_aux.append(lm.z)

            # Step 3: Padding - If only 1 hand, pad second hand with zeros to make 126 features
            if len(data_aux) == 63: # 21 landmarks * 3 = 63 for 1 hand
                data_aux.extend([0] * 63)

            # Step 4: Prediction
            if len(data_aux) == 126:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    probabilities = model.predict_proba([np.asarray(data_aux)])
                    confidence = np.max(probabilities) * 100

                    st.success(f"### Prediction: {prediction[0]}")
                    st.progress(int(confidence))
                    st.write(f"Confidence: {confidence:.1f}%")

                    # Debug panel to check model behavior
                    with st.expander("Debug - View all probabilities"):
                        st.write(f"Feature vector length: {len(data_aux)}")
                        prob_dict = dict(zip(model.classes_, probabilities[0]))
                        st.json(prob_dict)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.warning("No hand detected. Please bring hand closer in good lighting.")

    st.image(img, channels="BGR", caption="Landmark Detection")

st.success("Model Loaded: 126 features (normalized)")
