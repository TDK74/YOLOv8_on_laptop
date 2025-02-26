# YOLOv8_on_laptop
Quickly Run YOLOv8 for Real-Time Recognition on a Laptop.

* Based on the tutorial on this site:

<https://www.dfrobot.com/blog-13903.html>

I had no problems to deploy it on the CPU but I had some strugles and met defficulties to run it on CUDA (the graphical card).

# Setup Environment
* Laptop: Fujitsu Celcius H780
* Operating System: Windows 10 Pro x64
* Software: Python 3.8 (virtual env), 3.10, 3.11, 3.12 (Anaconda)
* Environment: Anaconda

My laptop has a NVIDIA Quadro P2000 and CUDA v6.1.

I couldn't find so old PyTorch wheel for my CUDA version (went back to wheels for CUDA v7.5).

See commands.txt and pip_freeze.txt for more details if interested.

## Note

Only YOLOv8 on a laptop for the moment.

Next will be Nvidia Jetson Nano (4 GB RAM) or Raspberry Pi 4B (2 GB RAM).
