"""Video transforms with OpenCV"""

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from sample_utils.turn import get_ice_servers

st.title("BeFit - Pushup Counter")

def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # prepare color
    img_color = cv2.pyrDown(cv2.pyrDown(img))
    for _ in range(6):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
    img_color = cv2.pyrUp(cv2.pyrUp(img_color))

    # prepare edges
    img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_edges = cv2.adaptiveThreshold(
        cv2.medianBlur(img_edges, 7),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        2,
    )
    img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

    # combine color and edges
    img = cv2.bitwise_and(img_color, img_edges)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="opencv-filter",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown(
    "This demo is based on "
    "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
    "Many thanks to the project."
)
