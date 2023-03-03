import sys
import os
# 工程化.
from pathlib import Path
FILE = Path(__file__).resolve() # 绝对路径.
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT) # runtime path in current path. 

print("run on:", ROOT)


# 允许的文件类型.
IMAGE_FILE = ['jpg','jpeg','png']
VIDEO_FILE = ['mp4','avi']

# 0: stop, 1: open detective 
STOP = 0

SAVE_PATH = "./file"
LOG_PATH = "./logs"

IS_REALTIME = True

# global IS_REALTIME