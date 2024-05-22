from pathlib import Path
import PIL
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

if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
    config_path = Path(settings.CONFIG_PATH)

try:
    # model = helper.load_model(config_path, model_path)
    model = helper.load_model('weights/best_3000_openvino_model/')
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
if source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)
else:
    st.error("Please select a valid source type!")
