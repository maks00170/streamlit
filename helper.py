from ultralytics import YOLO
import streamlit as st
import cv2
from mmdet.apis import DetInferencer
import PIL
import settings
import torch
import numpy as np

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def plot_box(result):
    image = result.orig_img
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.2
    fontColor = (0,0,255)
    thickness = 2
    lineType = 1 
    
    for idx, boxes in enumerate(result.boxes.xyxy):
        conf =  round(result.boxes.conf[idx].item(), 2)
        clas = result.names[result.boxes.cls[idx].item()]
        
        cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), color=(0, 0, 255), thickness=2)
        cv2.putText(image, f'{conf},{clas}', 
            (int(boxes[0]), int(boxes[1])), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

    return image

# ssh -p 2222  -i auth_key.pem -X echertova@cluster.hpc.hse.ru
# ssh -p 2222  -i auth_key.pem echertova@cluster.hpc.hse.ru
def load_model(model_path):
    model = YOLO(model_path)
    return model


def _display_detected_frames(model, st_frame, image, conf):
    image = cv2.resize(image, (720, int(720*(9/16))))

    res = model.predict(image, conf=conf, verbose=False)
    res_plotted = plot_box(res[0])
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

# def load_model(config_file, checkpoint_file):
#     print(config_file, checkpoint_file)
#     return DetInferencer(str(config_file), str(checkpoint_file), device=DEVICE, show_progress=False)


# def _display_detected_frames(model, st_frame, image, conf=0):
#     image = cv2.resize(image, (720, int(720 * (9 / 16))))

#     result = model(image, return_vis=True)
#     #print("result", result)
#     try:
#         visualized_image_array = result['visualization'][0]
#     except Exception as e:
#         visualized_image_array = np.array(image)

#     visualized_image = PIL.Image.fromarray(visualized_image_array)

#     st_frame.image(visualized_image, caption='Detected Video', channels="BGR", use_column_width=True)


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
