from ultralytics import YOLO
import streamlit as st
import cv2
import settings
import torch
import numpy as np

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

yolo_dont_work = False
frame_empty = 0
last_predict = [1]

# ssh -p 2222  -i auth_key.pem -X echertova@cluster.hpc.hse.ru
# ssh -p 2222  -i auth_key.pem echertova@cluster.hpc.hse.ru
def load_model(model_path):
    model = YOLO(model_path)
    return model


def _display_detected_frames(model, st_frame, image, conf):
    global frame_empty, last_predict

    image = cv2.resize(image, (720, int(720*(9/16))))
    
    if len(last_predict) == 0 and frame_empty < 5:
        res_plotted = image
        frame_empty += 1
        #print('dont work')
    else:
        res = model.predict(image, conf=conf, verbose=False)
        res_plotted = res[0].plot()
        last_predict = res[0].boxes.xyxy
        frame_empty = 0
        #print('work')
    
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                                             model,
                                             st_frame,
                                             image,
                                             conf,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))