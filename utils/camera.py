import time
import numpy as np
import cv2 as cv
from utils.base_camera import BaseCamera
import sys
import os
from PIL import Image
import math
from yolov5.yolo import YOLO
from utils import global_manager as gm

from config import *
import datetime


join = lambda *args: os.path.join(*args)

yolo = YOLO()


# gm.set_value("is_realtime", "None")


runtime_path = sys.path[0]


def edges_detection(frame):
    frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    guss_frame = cv.GaussianBlur(frame_grey, (5, 5), 0)

    edges = cv.Canny(guss_frame, 30, 100)  # 得到灰度图
    edges = np.expand_dims(edges, -1)
    edges = np.broadcast_to(edges, (frame.shape))

    return edges


def detecti_img(frame):
    start_time = time.time()
    # h, w, _ = frame.shape
    # print(h, w) # 480 640 in camera
    edges = edges_detection(frame)

    img = Image.fromarray(np.uint8(frame))

    img, result = yolo.detect_image(img)

    img_not_blur, _ = yolo.detect_image(
        Image.fromarray(np.uint8(frame)), is_blur=False)
    img_not_blur = np.array(img_not_blur)

    bottom_picture = np.hstack([edges, img_not_blur])

    detection_img = np.array(img)
    end_time = time.time()
    seconds = end_time - start_time
    fps = 1 / seconds

    result_image = np.full_like(img, 255)
    now = datetime.datetime.now()  # get current time.
    current_time = now.strftime("%Y-%m-%d %H:%M:%S").__str__()
    print(current_time)
    format_result = ""
    format_result_all = ""  # reset

    if result:

        for index, line in enumerate(zip(*result)):
            label = yolo.class_names[int(line[0])]
            conf = round(line[1], 2)  # confidentce.
            top, left, bottom, right = line[2]
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(img.size[1], np.floor(bottom).astype('int32'))
            right = min(img.size[0], np.floor(right).astype('int32'))

            centre_x = (right + left) / 2
            centre_y = (bottom + top) / 2

            substance_h = int(bottom - top)
            substance_w = int(right - left)
            coordinate = tuple((centre_x, centre_y))
            size = tuple((substance_w, substance_h))

            format_result = index.__str__() + " " + label.__str__()+" "+conf.__str__() + \
                " "+coordinate.__str__() + " " + size.__str__()
            format_result_all += format_result+'\n'
            cv.putText(result_image, f"{format_result}", (1,
                       25*(index+1)), cv.FONT_ITALIC, 0.75, (0, 0, 0), 1)
        print(format_result_all)

    print(current_time.split(" ")[0])
    format_write = current_time + ":" + "\n" + format_result_all + \
        "\n" + "===============================" + "\n"

    with open(join(runtime_path, "logs", current_time.split(" ")[0]+".log"), 'a') as f:
        f.write(format_write)

    frame = np.hstack([frame, detection_img])
    frame = np.vstack([frame, bottom_picture])

    cv.putText(frame, f"FPS:{fps.__str__():4.4}", (5, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return frame, result_image


class Camera(BaseCamera):

    def __init__(self):
        super(Camera, self).__init__()

    @staticmethod
    def frames():  # 重写BaseCamera的frames类

        if gm.get_value("is_realtime"):
            print("realtime mode")
            cap = cv.VideoCapture(0)  # get local machine camera.

            if not cap.isOpened():
                return RuntimeError("can't open local camera!")
            # cap.set(3, 720)
            # cap.set(4, 1280)

            # while True:
            #     res, frame = cap.read() # read.

            #     if res: # is have image.
            #         start_time = time.time()
            #         img = Image.fromarray(np.uint8(frame))
            #         img, result = yolo.detect_image(img)
            #         detection_img = np.array(img)
            #         end_time = time.time()
            #         seconds = end_time - start_time
            #         fps = 1 / seconds

            #         result_image = np.full_like(img, 255)

            #         if result:
            #             format_result = "" # reset
            #             # print(result)

            #             for index, line in enumerate(zip(*result)):
            #                 label = yolo.class_names[int(line[0])]
            #                 conf = round(line[1], 2)
            #                 top, left, bottom, right = line[2]
            #                 top     = max(0, np.floor(top).astype('int32'))
            #                 left    = max(0, np.floor(left).astype('int32'))
            #                 bottom  = min(img.size[1], np.floor(bottom).astype('int32'))
            #                 right   = min(img.size[0], np.floor(right).astype('int32'))
            #                 centre_x = (bottom - top) / 2
            #                 centre_y = (right - left) / 2
            #                 coordinate = tuple((centre_x, centre_y))

            #                 format_result = label.__str__()+" "+conf.__str__()+" "+coordinate.__str__()
            #                 cv.putText(result_image, f"{format_result}", (1, 20*(index+1)), cv.FONT_ITALIC, 0.7, (0, 0, 0), 1)

            #             print(format_result)
            #             # cv.putText(result_image, f"{format_result}", (1, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            #         frame = np.hstack([frame, detection_img])

            #         cv.putText(frame, f"FPS:{fps.__str__():4.4}", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            #         yield cv.imencode('.jpg', frame)[1].tobytes(), cv.imencode('.jpg', result_image)[1].tobytes() # to bytes.
            #         cv.waitKey(44)

            #     else:
            #         print("can't get image, please check you camera if opened?")
            #         break

            # cap.release() # release resource.

        elif not gm.get_value("isrealtime"):
            print("Video mode.")
            with open(join(SAVE_PATH, "./file_name.txt"), "r") as f:
                file_name = f.readline()
            print(file_name+"has been download in local!")

            frame_counter = 0

            # real file name.
            cap = cv.VideoCapture(join(SAVE_PATH, file_name))
            if not cap.isOpened():
                return RuntimeError(f"can't open {file_name} Video")

        else:
            print("not uninitialized")

        while True:
            # now = datetime.datetime.now() # get current time.
            # current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            res, frame = cap.read()
            if res:
                if not gm.get_value("isrealtime"):
                    frame_counter = 1
                    if frame_counter == int(cap.get(cv.CAP_PROP_FRAME_COUNT)):  # reset
                        frame_counter = 0  # reset.
                        cap.set(cv.CAP_PROP_POS_FRAMES, frame_counter)

                frame, result_image = detecti_img(frame)
                # to bytes.
                yield cv.imencode('.jpg', frame)[1].tobytes(), cv.imencode('.jpg', result_image)[1].tobytes()

            else:
                continue

            cv.waitKey(50)
        cap.release()
