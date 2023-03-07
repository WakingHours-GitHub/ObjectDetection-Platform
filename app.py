from flask import Flask, request, render_template, Response
import cv2 as cv
from PIL import Image
import numpy as np
from config import *
import os
import sys
import base64
from utils.camera import Camera

# yolov5版本
from yolov5.yolo import YOLO


# 宏定义.
join = lambda *args: os.path.join(*args)

app = Flask(__name__, static_folder="./static", template_folder="templates")



yolo = YOLO()

@app.route("/image", methods=["POST", ""])
def parse_image():
    image = request.files["image"] # 从form表单种, 拿出image的input.
    image_name = image.filename
    file_path = os.path.join(SAVE_PATH, image_name)
    image.save(file_path)
    print(image_name, "has been download in local")
    if image_name.split(".")[-1] in IMAGE_FILE: # 是图片
        source_img = cv.imread(file_path) # read image
        img = Image.fromarray(np.uint8(source_img)) # 转换为Image. 
        img_torch, box_information = yolo.detect_image(img)
        print(box_information)
        img = np.array(img_torch)
        img = np.hstack([source_img, img])
        cv.imwrite(file_path, img)

        # _, img_encoded = cv.imencode('.jpg', img)

        try:
            img_stream = ''
            with open(file_path, 'rb') as img_f:
                img_stream = img_f.read()
                img_stream = base64.b64encode(img_stream).decode()
            return render_template('./image_process.html', image_url=img_stream)
        except:
            print("image parse faild")
            return render_template('./image_process.html')



        # response = img_encoded.tobytes()
        # return Response(response=response, status=200, mimetype='image/jpg')

    elif image_name.split(".")[-1] in VIDEO_FILE: # 视频
        global IS_REALTIME
        IS_REALTIME = False # 是视频。 
        print("%s Video uploading, please wait. " % image_name)
        with open(join(SAVE_PATH, "./file_name.txt"), "w") as f_writer:
            f_writer.write(image_name) # 将文件名字写到文件里
        
        return render_template("video_process.html") # 返回。




    return render_template("./index.html")



@app.route("/")
def main_page():
    return render_template("./index.html")


@app.route("/realtime")
def realtime():
    global IS_REALTIME
    IS_REALTIME = True
    return render_template("video_process.html")



@app.route("/architecture")
def architecure():
    
    return render_template("./architecutre.html")



def gen(camera):
    """Video streaming generator function."""
    while True:
        frame, result = camera.get_frame()
        # frame = camera.get_frame()
        yield (b'--frame\r\n'
                b''b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_text(camera):
    while True:
        _, frame = camera.get_frame()
        yield (b'--frame\r\n'
                b''b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# url_for(video_feed) # 找到这个函数, 返回response(gen产生的yield.).
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(
        gen(Camera()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/text_feed')
def text_feed():
    return Response(
        gen_text(Camera()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )



# check file if exist.
def init():
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)

    for ele in os.listdir(SAVE_PATH):
        ele_path = os.path.join(SAVE_PATH, ele)
        os.remove(ele_path)
    print("file dir has been empty")
    
    





# run 
if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", port=5000, debug=True)