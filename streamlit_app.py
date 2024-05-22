import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import av
import numpy as np
import collections
import requests

st.title("Hand gesture detection")
st.write("This app uses MediaPipe to estimate hand pose in real-time.")

with st.sidebar:
    st.title("設定")
    num_hands = st.slider("検出する手の最大数", min_value=1, max_value=10, value=3, step=1)
    thickness = st.slider("エッジの太さ", min_value=1, max_value=10, value=2, step=1)
    circle_radius = st.slider("ノードの大きさ", min_value=1, max_value=10, value=2, step=1)
    min_detection_confidence = st.slider("検出閾値", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    min_tracking_confidence = st.slider("トラッキング閾値", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    size_font = st.slider("フォントサイズ", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    color_edge = st.color_picker("エッジ色", "#00bfff")
    color_node = st.color_picker("ノード色", "#0000cd")
    color_font = st.color_picker("フォント色", "#c71585")

# url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
# r = requests.get(url)

# base_options = python.BaseOptions(model_asset_buffer=r.content)

base_options = python.BaseOptions(model_asset_path="gesture_recognizer_20240522.task")
options = vision.GestureRecognizerOptions(
                                        base_options=base_options,
                                        num_hands=num_hands,
                                        min_hand_detection_confidence=min_detection_confidence,
                                        min_tracking_confidence=min_tracking_confidence,
                                        )
recognizer = vision.GestureRecognizer.create_from_options(options)

def hex_to_rgb(color_hex):
    Color_rgb = collections.namedtuple("Color_rgb", ["r", "g", "b"])
    return Color_rgb(*[int(color_hex.replace("#", "")[i:i+2], 16) for i in (0, 2, 4)])


def calc_bounding_box(image_cv2, hand_landmarks_gesture):
    image_width, image_height = image_cv2.shape[1], image_cv2.shape[0]

    landmark_array = np.empty((0, 2), int)

    for landmark in hand_landmarks_gesture:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    Bbox = collections.namedtuple("Bbox", ["x", "y", "w", "h"])

    return Bbox(*cv2.boundingRect(landmark_array))


color_edge_rgb = hex_to_rgb(color_edge)
color_node_rgb = hex_to_rgb(color_node)
color_font_rgb = hex_to_rgb(color_font)

def callback(frame):
    mp_drawing = mp.solutions.drawing_utils
    
    # Load the input image.
    image = frame.to_ndarray(format="bgr24")

    result_recognition = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))

    if result_recognition.hand_landmarks:
        for hand_landmarks, gestures in zip(result_recognition.hand_landmarks, result_recognition.gestures):

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            mp_drawing.draw_landmarks(
                image, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(color_node_rgb.b, color_node_rgb.g, color_node_rgb.r), thickness=thickness, circle_radius=circle_radius),
                mp_drawing.DrawingSpec(color=(color_edge_rgb.b, color_edge_rgb.g, color_edge_rgb.r), thickness=thickness, circle_radius=circle_radius),
            )

            bbox = calc_bounding_box(image, hand_landmarks)

            cv2.putText(image, gestures[0].category_name, (bbox.x, bbox.y), cv2.FONT_HERSHEY_DUPLEX, size_font, (color_font_rgb.b, color_font_rgb.g, color_font_rgb.r))


    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(key="hand-gesture", video_frame_callback=callback, async_processing=True, media_stream_constraints={"video": True, "audio": False}, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
