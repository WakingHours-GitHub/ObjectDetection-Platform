import time
import numpy as np
import cv2 as cv
from utils.base_camera import BaseCamera
import sys
import os
from PIL import Image

# from app import YOLO
from yolov5.yolo import YOLO

yolo = YOLO()



# global_manager.set_value("real_time_type", "None")


runtime_path = sys.path[0]

class Camera(BaseCamera):

    def __init__(self):
        super(Camera, self).__init__()
        # self.yolo = yolo



    @staticmethod
    def frames(): # 重写BaseCamera的frames类
        print("realtime mode")
        cap = cv.VideoCapture(0) # get local machine camera.
        
        if not cap.isOpened():
            return RuntimeError("can't open local camera!")
        
        while True:
            res, frame = cap.read() # read.

            if res: # is have image. 
                start_time = time.time()
                img = Image.fromarray(np.uint8(frame))
                img, result = yolo.detect_image(img)
                detection_img = np.array(img)
                end_time = time.time()
                seconds = end_time - start_time
                fps = 1 / seconds

                print(fps)
                frame = np.hstack([frame, detection_img])
                cv.putText(frame, f"FPS:{fps.__str__():4.4}", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)



                yield cv.imencode('.jpg', frame)[1].tobytes() # to bytes.
                cv.waitKey(50)

            else:
                print("can't get image, please check you camera if opened?")
                break

        cap.release()

