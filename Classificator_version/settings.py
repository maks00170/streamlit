from pathlib import Path
import sys

file_path = Path(__file__).resolve()

root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

# Sources
WEBCAM = 'Webcam'

SOURCES_LIST = [WEBCAM]


DETECTION_MODEL = ROOT / ('weights/rtmdet_best_coco_bbox_mAP_epoch_116.pth')
                          # weights
                          # 'best_coco_bbox_mAP_epoch_34.pth')
                          # 'best_3000.pt')
CONFIG_PATH = (ROOT / ('configs/rtmdet_tiny_8xb32-300e_coco_custom.py'))
               # ('configs/retinanet_r18_fpn_1x_coco_custom.py'))
# Webcam
WEBCAM_PATH = 0