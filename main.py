from ultralytics import YOLO
import os
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["WANDB_MODE"] = "offline"
if __name__ == '__main__':
    # model = (YOLO("ultralytics/cfg/models/v8/yolov8.yaml"))  # 模型 CA 4
    model = (YOLO("BRSTD/VIT_bl.yaml"))  # 模型 CA
    model.train(**{'cfg': 'ultralytics/cfg/default.yaml'})  # 配置
#     model = YOLO(r'runs/detect/DOTA/weights/best.pt')  # load a partially trained model

# # Resume training
#     results = model.train(resume=True)

