import time
import numpy as np
import cv2 as cv
from utils.base_camera import BaseCamera
import sys
import os
from PIL import Image

from yolov5.yolo import YOLO
from yolov5 import *
from unet.unet import Unet
from unet import *

from utils import global_manager



# global_manager.set_value("real_time_type", "None")


runtime_path = sys.path[0]
yolo = YOLO()
unet = Unet()

class Camera(BaseCamera):

    def __init__(self):
        super(Camera, self).__init__()



    @staticmethod
    def frames(): # 重写BaseCamera的frames类
        real_time_type = global_manager.get_value("real_time_type")
        is_stop = global_manager.get_value("is_stop")

        print("camera: real_time_type", real_time_type)

        if "video" == real_time_type:
            print("video")
            with open(os.path.join(runtime_path, "./file_name.txt"), "r") as f:
                file_name = f.readline()
                
            print(file_name)
            
            cap = cv.VideoCapture(os.path.join("file", file_name))

            frame_counter = 0

            if not cap.isOpened():
                return RuntimeError("could not open camera!")

            while True:
                if is_stop == 'yes':
                    break
                res, frame = cap.read()
                # frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

                """
                        h, w, _ = source_img.shape
            if h > 2000 or w > 2000:
                h = h // 2
                w = w // 2
                source_img = cv.resize(source_img, (int(w), int(h)))
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = Image.fromarray(np.uint8(source_img)) # 转换为Image. 
            img = np.array(yolo.detect_image(img))"""

                if res:
                    frame_counter += 1
                    if frame_counter == int(cap.get(cv.CAP_PROP_FRAME_COUNT)): # reset
                        frame_counter = 0
                        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

                    h, w, _ = frame.shape
                    img = cv.GaussianBlur(frame, (7, 7), 0)
                    # img = frame
                    if h > 2000 or w > 2000:
                        h = h // 2
                        w = w // 2
                        img = cv.resize(img, (int(w), int(h)))
                    Image_img = Image.fromarray(np.uint8(img)) # 转换为Image. 
                    segmentation_img = np.array(unet.detect_image(Image_img))
                    
                    detection_img = np.array(yolo.detect_image(Image_img))

                    yield cv.imencode('.jpg', np.hstack([frame, segmentation_img, detection_img]))[1].tobytes() # to bytes.
                else:
                    print("not get cap, please check you camera is opened!")
                
                cv.waitKey(50)


        elif "realtime" == real_time_type: #

            print("realtime")
            cap = cv.VideoCapture(0)

            if not cap.isOpened():
                return RuntimeError("could not open camera!")
            while True:
                if is_stop == 'yes':
                    break
                
                res, frame = cap.read()

                if res:
                    start_time = time.time()
                    img = Image.fromarray(np.uint8(frame))
                    segmentation_img = np.array(unet.detect_image(img))
                    detection_img = np.array(yolo.detect_image(img))
        
                    # detection_img_mask = np.array(yolo.detect_image(segmentation_img*np.array(Image_img)))

                    end_time = time.time()
                    seconds = end_time - start_time
                    fps = 1 / seconds
                    print(fps)
                    cv.putText(frame, f"FPS:{fps.__str__():4.4}", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    yield cv.imencode('.jpg', np.hstack([frame, segmentation_img, detection_img]))[1].tobytes() # to bytes.
                else:
                    print("not get cap, please check you camera is opened!")
                cv.waitKey(50)

        cap.release()
        


