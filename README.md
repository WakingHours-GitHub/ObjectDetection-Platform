# Surface Water Floating Detection Platform 🧐

This is an amazing object detection platform, primarily focused on detecting floating objects on water surfaces. You can use it to detect anything by replacing the dataset with your own task-specific data!
<br>
This platform identifies and classifies various types of floating objects in real-time, while Flask provides a user-friendly interface for easy interaction with the platform. (•̃ ᴗ•̃)
<br>
The platform is based on the YOLOv5🚀 object detection algorithm and Flask, a web framework. 🎨

---

## Update (2026 Feb)

Due to the high cost of maintaining the server and limited resources, we regret to decide that the website will be taken offline. We sincerely thank you for your support and interest in this project over the years. 🙏

Although the website will no longer be available, the source code and resources will remain accessible on GitHub for anyone who wishes to explore or continue the project.

Main website (archived):
![Main Website](https://github.com/WakingHours-GitHub/ObjectDetection-Platform/blob/master/resources_md/main.png)
Detected surface on water:
![Detected Surface](https://github.com/WakingHours-GitHub/surface-water-floating-detection-platform/blob/master/resources_md/detected.png)

For any inquiries or further collaboration, feel free to reach out via GitHub or email. Thank you for being part of this journey!

## Architecture
![Architecture](https://github.com/WakingHours-GitHub/surface-water-floating-detection-platform/blob/master/static/architecutre.svg)

---

## Getting Started ☜(ˆ▿ˆc)

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes. See the deployment section for notes on how to deploy the project on a live system.

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Run as Test:
```bash
python3 -u app.py
```

### Run as Server:
```bash
nohup python3 -u app.py > run.log 2>&1 &
```

For deployment, consider using Nginx. Refer to relevant blogs or documentation for guidance.

---

## Deployment

Detailed deployment instructions will be added soon.

---

## Built With

- [Flask](http://flask.pocoo.org/) - The web framework used
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [YOLOv5 (Ultralytics Edition)](https://github.com/ultralytics/yolov5) - YOLOv5 implementation by Ultralytics
- [YOLOv5 (Bubbliiiing Edition)](https://github.com/bubbliiiing/yolov5-pytorch) - Another YOLOv5 implementation by Bubbliiiing

---

## Authors

All of this is created by me (─‿‿─):

- **Waking Hours** - *Coder* - [GitHub](https://github.com/WakingHours-GitHub)
- **Waking Hours** - *Artist* - [GitHub](https://github.com/WakingHours-GitHub)

---

## Acknowledgments

- Thanks to myself for the hard work.
- Special thanks to my friend Hongbo Wang, who helped collect samples of floating objects and created datasets using [LabelImg](https://github.com/heartexlabs/labelImg).
- Gratitude to everyone contributing to the fight against the epidemic!
- And many others!

