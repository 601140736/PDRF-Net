import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
from ultralytics import YOLO




def main():
    # ultralytics/cfg/models/v8/yolov8s-twoCSP-fusion-enhance.yaml
    model = YOLO('dual_cfg/yolov8_sp_dual_pre.yaml')
    model.load('runs/FLIR/train8/weights/best.pt')

    model.info()

    results = model.train(data='data/FLIR.yaml',
                        epochs=300,
                        imgsz=640,
                        workers=24,
                        batch=24,
                        device='0,1,2',
                        amp=True,
                        project='runs/',
                        name='FLIR/train',
                        )



if __name__ == "__main__":
    main()