🌊 Underwater Trash Plastic Detection 🛢️
Table of Contents
🌟 About The Project
This is an underwater plastic trash detection and analysis system built to address marine pollution. We have developed this project using the latest advancements in deep learning, specifically by training custom datasets with YOLOv5 and YOLOv8 architectures. This system includes two distinct models: one based on YOLOv5 for high-speed object detection (identifying trash locations) and a separate model using YOLOv8 for high-precision instance segmentation (distinguishing individual plastic items and mapping their exact boundaries)

Built With
YOLOv5
YOLOv8
🚀 Getting Started
We utilized Google Colab with GPU support to run our YOLOv5 model.

Prerequisites
Python
Installation for YOLOv5 object detection
Clone the repository
!git clone https://github.com/ultralytics/yolov5.git
Install required dependencies
!pip install -r requirements.txt
Run the YOLOv5 object detection model
!python detect.py --weights bestpla.pt --source path/to/folder/orImage
Installation for YOLOv8 instance segmentation
Install ultralytics
!pip install ultralytics
Install required dependencies
!pip install -r requirements.txt
Run the YOLOv5 object detection model
!yolo predict model='runs/segment/yolov8n-seg/weights/best.pt' source='detection source file' name='folder_name'
🎯 Usage
This project serves to detect underwater garbage, including items like plastic bags. It can contribute to ocean cleanup efforts and environmental monitoring. To run the project, consider Google Colab.

📊 Output
The images above depict the output of our underwater trash plastic detection models. The model successfully identifies and outlines plastic waste items, such as plastic bags, cups, metal_can and cups, contributing to efforts to clean up our oceans.

Underwater waste detection image Underwater waste detection image Underwater waste detection image Underwater waste detection image Underwater waste detection image Underwater waste detection image
