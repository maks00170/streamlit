from ultralytics import YOLO
import streamlit as st
import cv2
import settings
import torch
import copy

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

yolo_dont_work = True

def plot_box(result):
    global yolo_dont_work
    image = result.orig_img
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.2
    fontColor = (0,0,255)
    thickness = 2
    lineType = 1 
    if len(result.boxes.xyxy) == 0:
        yolo_dont_work = True
    for idx, boxes in enumerate(result.boxes.xyxy):
        conf =  round(result.boxes.conf[idx].item(), 2)
        clas = result.names[result.boxes.cls[idx].item()]
        yolo_dont_work = False
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


def _display_detected_frames(model, model_class, st_frame, image, conf):
    im = copy.deepcopy(image)
    image = cv2.resize(image, (720, int(720*(9/16))))

    if yolo_dont_work:
        image1 = cv2.resize(im, (240, 160))
        score = model_class(image1).detach().item()
        print(score)
        if score > 0.9:
            print('Yolo work')
            res = model.predict(image, conf=conf, verbose=False)
            res_plotted = plot_box(res[0])
        else:
            print('Yolo not work')
            res_plotted = image
    else:
        print('class not work')
        image = cv2.resize(image, (720, int(720*(9/16))))
        res = model.predict(image, conf=conf, verbose=False)
        res_plotted = plot_box(res[0])

    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_webcam(conf, model, model_class):
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
                                             model_class,
                                             st_frame,
                                             image,
                                             conf,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))