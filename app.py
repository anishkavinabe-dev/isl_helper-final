import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib

st.set_page_config(page_title="ISL LIVE Translator")
st.title("🤟 LIVE ISL Translator")
st.write("Take photo - instant prediction")

@st.cache_resource
def load_model():
    return joblib.load('isl_model.pkl')
model = load_model()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

img_file = st.camera_input("Show your ISL sign")

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        results = hands.process(img_rgb)
        data_aux = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x, lm.y, lm.z])
            if len(data_aux) == 63:
                data_aux.extend([0]*63)
            if len(data_aux) == 126:
                pred = model.predict([np.asarray(data_aux)])
                st.success(f"Prediction: {pred[0]}")
        else:
            st.warning("No hand detected")
    st.image(img, channels="BGR")

st.success("Model loaded: 126 features")
