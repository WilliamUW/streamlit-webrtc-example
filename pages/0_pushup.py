import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration
import PoseModule as pm

st.title("BeFit - Pushup Counter")

# Initialize the pose detector
detector = pm.poseDetector()

# Define the callback function for video frame processing
def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Process the image with the pose detector
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # Example of calculating the angle between three points
        elbow = detector.findAngle(img, 11, 13, 15)
        shoulder = detector.findAngle(img, 13, 11, 23)
        hip = detector.findAngle(img, 11, 23, 25)

        # Simulate updating pushup count and feedback (replace with your logic)
        # This is just a placeholder to show where you would implement your logic
        # You need to replace it with your actual pushup detection and counting logic
        # if elbow > 160:
        #     st.session_state.count += 1  # Increment the count

    # Return the processed video frame
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


# Initialize session state for pushup count if it doesn't exist
if 'count' not in st.session_state:
    st.session_state.count = 0

# Streamlit-WebRTC component to process video stream
webrtc_streamer(
    key="pushup_detector",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,  # Use the RTC configuration for ICE servers
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Display the pushup count
st.write(f"Pushup Count: {st.session_state.count}")

