from pathlib import Path
import streamlit as st
import settings
import helper

st.set_page_config(
    page_title="Gestures detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Gestures detection")
st.sidebar.header("ML Model Config")
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100



try:
    # model = helper.load_model(config_path, model_path)
    model = helper.load_model('weights/best_3000_openvino_model/')
except Exception as ex:
    st.error(ex)

st.sidebar.header("Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
if source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)
else:
    st.error("Please select a valid source type!")