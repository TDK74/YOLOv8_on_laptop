# YOLOv8_on_laptop
Quickly Run YOLOv8 for Real-Time Recognition on a Laptop.

* Based on the tutorial on this site:

<https://www.dfrobot.com/blog-13903.html>

* A Comprehensive and Detailed Introduction to YOLOv8:

<https://www.dfrobot.com/blog-13844.html>

## 

# Setup Environment
* Laptop: Fujitsu Celcius H780
* Operating System: Windows 10 Pro x64
* Software: Python 3.8 (virtual env - workspace), 3.10, 3.11, 3.12 (Anaconda)
* Environment: Anaconda
* Object detection models: YOLOv8n, YOLOv8s, YOLOv8m
* Segmentation models: YOLOv8n-seg, YOLOv8s-seg, YOLOv8m-seg
* Pose detection models: YOLOv8n-pos, YOLOv8s-pos, YOLOv8m-pos

My laptop has a NVIDIA Quadro P2000 and CUDA v6.1.

There were no problems to deploy the models on the CPU, but I had some struggles to run them on CUDA (the graphical card).

I couldn't find so old PyTorch wheel for my CUDA version (went back to wheels for CUDA v7.5).

See **_commands.txt_** and **_pip_freeze.txt_** for more details if interested.

Certain refactoring of the code was done by me.

## Note

The branches will be left unmerged to the master in order to get the files from Objects Detection, Segmentation and Pose Detection easily distinguished.

Only YOLOv8 on a laptop for the moment.

Next will be on Nvidia Jetson Nano (4 GB RAM) or on Raspberry Pi 4B (2 GB RAM).
