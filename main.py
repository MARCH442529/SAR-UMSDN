from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    ############## 这是train的代码 ##############
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8n.yaml")  # 初始化模型
    model = YOLO(r"ultralytics/cfg/models/v8/yolov8-seg.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-CTF.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-CTF-CFE.yaml")  # 初始化模型

    model.train(data=r"F:/JD/ultralytics/Datasets/tree/data.yaml", model='',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=1,
                device='0',
                pretrained=False,
                optimizer='AdamW',  # using SGD
                # resume='', # last.pt path
                amp=False,  # close am
                # fraction=0.2,
                project='runs',
                name='yolov8-seg',
                visualize=True,

                )  # 训练

    ############## 这是val和predict的代码 ##############
    # model = YOLO(r"./best.pt")
    # model.val(data=r"ultralytics/cfg/datasets/mydata.yaml", batch=1, save_json=True, save_txt=False)  # 验证
    # model.predict(source=r"Datasets/llvip/images/test", save=True)  #   检测
