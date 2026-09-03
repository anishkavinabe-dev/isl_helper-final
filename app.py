import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="ISL LIVE Translator", layout="centered")
st.title("🤟 LIVE ISL Translator")
st.write("LIVE camera - No photo needed")

@st.cache_resource
def load_model():
    return joblib.load('isl_model.pkl')

model = load_model()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Safe STUN/TURN - no private credentials
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun.l.google.com:5349"]},
        {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"},
        {"urls": ["turn:openrelay.metered.ca:443"], "username": "openrelayproject", "credential": "openrelayproject"},
    ]
})

class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.last_pred = "Show Sign"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        data_aux = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x, lm.y, lm.z])

            # Pad if 1 hand only -> make 126
            if len(data_aux) == 63:
                data_aux.extend([0]*63)

            if len(data_aux) == 126:
                try:
                    pred = model.predict([np.asarray(data_aux)])
                    self.last_pred = str(pred[0])
                except Exception:
                    pass

            cv2.rectangle(img, (0,0), (350, 70), (0,0,0), -1)
            cv2.putText(img, self.last_pred, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        else:
            cv2.putText(img, self.last_pred, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="isl-live-final",
    video_processor_factory=HandProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)

st.success("Model loaded: 126 features")
