# Object Detection System on Raspberry Pi

本科毕业设计，一个部署到树莓派上的轻量化目标检测系统平台。

## Overview

This project provides an object detection system running on a Raspberry Pi.

It supports multiple input modes (image, video, and camera) and allows users to select different models for detection.

The system also exports detection performance (FPS) results into an Excel file for further analysis.

<img src=".\assets\功能模块图.jpg" alt="System Architecture (Modules)" style="zoom: 33%;" />

## System Workflow

1. Run `main.py` on the Raspberry Pi environment.
2. Launch the **Object Detection System Platform**.
3. Select the detection model and input source.
4. Perform object detection on the chosen input (image, video, or camera).
5. Export detection results (FPS) to an Excel file in the project directory.

<img src=".\assets\流程图.jpg" alt="pipeline" style="zoom: 33%;" />

## Usage Guide

<img src=".\assets\项目结构.jpg" alt="Project Structure" style="zoom: 33%;" />

1. **Run on Raspberry Pi**：Open the terminal and navigate to the project directory

   ```shell
   cd /home/pi/your_project
   ```

   Run the main program:

   ```shell
   python3 main.py
   ```

   

2. **Features of the Object Detection Platform**

   - **Main Interface**：Provides access to model selection and input source configuration.

     <img src=".\assets\主界面.jpg" alt="Main Interface" style="zoom:33%;" />

   - **Model Selection**：

     - Choose between available object detection models (e.g., YOLO series or other integrated models).
     - The system automatically loads the corresponding weights.

     <img src=".\assets\模型文件.jpg" alt="Model Files" style="zoom:33%;" />

     <img src=".\assets\模型选择.jpg" alt="Model Selection" style="zoom:33%;" />

   - **Image Input**：

     - Upload single image.
   
       <img src=".\assets\图片输入.jpg" alt="Image Input" style="zoom:33%;" />
   
     - The system performs object detection and displays the results.
   
       <img src=".\assets\图片输入2.jpg" alt="Detect Result" style="zoom:33%;" />

   - **Video Input**：

     - Select a local video file.
   
       <img src=".\assets\视频输入.jpg" alt="Video Input" style="zoom:33%;" />
   
     - The system processes the video frame by frame with detection results shown in real time.
   
       <img src=".\assets\视频输入2.jpg" alt="Detect Result" style="zoom: 33%;" />
   
   - **Camera Input**：
   
     - Supports Raspberry Pi’s built-in camera or external USB cameras.
   
     - Enables real-time object detection from the live camera stream.
   
       <img src=".\assets\摄像头输入.jpg" alt="Camera Input" style="zoom:33%;" />
   
   - **Result Export**：
   
     - The system records the detection FPS during runtime.
     
     - After detection, the FPS results are automatically exported into an **Excel file**.
     
     - The file is stored in the project directory for later analysis.
     
       <img src=".\assets\结果导出图.jpg" alt="FPS Result" style="zoom:50%;" />
