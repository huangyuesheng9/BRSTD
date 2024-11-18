from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    # Load a model
    # model = YOLO('runs\detect/visdrone_n\weights/best.pt')  # load an official model
    model = YOLO(r'E:\第二篇结果\best.pt')
    metrics = model.val(data='data/AI-TOD.yaml', iou=0.7, conf=0.001, half=False, device=0, save_json=True, name=r'val\AI-TOD_n')
    # Validate the model
    # metrics = model.val(data='ultralytics/datasets/coco.yaml', iou=0.7, conf=0.001, half=False, device=0, save_json=True)