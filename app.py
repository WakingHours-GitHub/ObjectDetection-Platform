from flask import Flask, request, render_template, Response
import cv2 as cv
from PIL import Image
import numpy as np
from config import *
import os
import sys


# yolov5版本
from yolov5.yolo import YOLO


# 宏定义.
join = lambda *args: os.path.join(*args)

app = Flask(__name__, static_folder="./static", template_folder="templates")



yolo = YOLO()

@app.route("/image", methods=["POST"])
def parse_image():
    image = request.files["image"] # 从form表单种, 拿出image的input.
    image_name = image.filename
    file_path = os.path.join(SAVE_PATH, image_name)
    image.save(file_path)
    print(image_name, "has been download in local")
    if image_name.split(".")[-1] in IMAGE_FILE: # 是图片
        source_img = cv.imread(file_path) # read image
        img = Image.fromarray(np.uint8(source_img)) # 转换为Image. 
        
        img = np.array(yolo.detect_image(img))
        _, img_encoded = cv.imencode('.jpg', img)
        response = img_encoded.tobytes()
        return Response(response=response, status=200, mimetype='image/jpg')




    return render_template("./index.html")



@app.route("/")
def main_page():
    
    return render_template("./index.html")





@app.route("/architecture")
def architecure():
    return render_template("./architecutre.html")





# check file if exist.
def init():
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    for ele in os.listdir(SAVE_PATH):
        ele_path = os.path.join(SAVE_PATH, ele)
        os.remove(ele_path)
    print("file dir has been empty")
    
    





# run 
if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", port=5000, debug=True)