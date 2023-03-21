import smtplib
from email.mime.text import MIMEText

import datetime
import sys
import os
# from config import ROOT, LOG_PATH

ROOT = "/home/wakinghours/programming/floating-detection-platform"
LOG_PATH = "logs"

join = os.path.join

NOW = datetime.datetime.now()
current_time = NOW.strftime("%Y-%m-%d %H:%M:%S").__str__()
today = current_time.split(' ')[0]


SENDER = "fwjhuc@outlook.com"  # 发送邮箱
with open(os.path.join(ROOT, "utils", "passwd")) as f:
    PASSWD = f.read()  # 读取密码.

RECEIVER = ["WakingHoursHUC@outlook.com"]  # 接收邮箱

def send_mail(mode=0):
    with open(join(ROOT, LOG_PATH, today + ".log"), "r") as fr:
        log_list = fr.readlines()
    with open(join(ROOT, "utils", "template.html"), "r") as fr:
        template = fr.read()
    template = template.replace("{{TODAY}}", today).replace("{{LOG_CONTENT}}", "<br>".join(log_list))
    # template

    print(template)
    
    email = MIMEText(template,'html','utf-8')
    email["Subject"] = today + ": LOG"
    email['From'] = SENDER 
    email['TO'] = ",".join(RECEIVER)

    with smtplib.SMTP("smtp.office365.com", 587) as smtp:
        smtp.ehlo()  # 向服务器标识用户身份
        smtp.starttls()
        smtp.login(SENDER, PASSWD)  # 登录
        smtp.send_message(email)  # 传入创建好的对象, 进行发送

    print("send done!")



    




    



def main() -> None:
    NOW = datetime.datetime.now()
    current_time = NOW.strftime("%Y-%m-%d %H:%M:%S").__str__()
    today = current_time.split(' ')[0]


    today_list = os.listdir(join(ROOT, LOG_PATH))
    today_list.sort()

    print(today_list[-1])




if __name__ == "__main__":
    # main()
    send_mail()
