
from ultralytics import YOLO
model= YOLO(r"runs/FLIR/train51/weights/best.pt")
model(source='/root/sp-base-dual/datasets/FLIR/images/test',save=True,name='output', visualize=True)
