# Hand Gesture Recognition for IoT Control

> A Deep Learning-powered system that translates real-time hand signs into actionable commands for IoT devices, utilizing MobileNetV2 for fast and efficient edge processing.

## Overview

As smart environments become more prevalent, the need for seamless, touchless interaction grows. This project addresses that need by using deep learning to recognize specific hand movements as distinct commands. By translating these physical gestures into digital signals, users can control and customize tasks for connected IoT devices entirely hands-free.

## System Architecture

The recognition pipeline is designed for continuous, real-time video processing:

1. **Capture Gesture:** The system continuously captures video input via a camera.
2. **Process Gesture Data:** The raw frames are passed through MediaPipe and TensorFlow for hand tracking and classification.
3. **Decision Logic:** The system evaluates if a registered gesture is recognized.
    * **If Yes:** The designated command is triggered on the target IoT device, and the process repeats.
    * **If No:** The system ignores the frame and continues capturing new input.

## Model & Deep Learning

This project uses a Convolutional Neural Network (CNN) architecture, specifically **MobileNetV2**. 

* **Why MobileNetV2?** It is highly optimized for speed and computational efficiency, making it ideal for processing video frames in real-time and deploying on lightweight or edge hardware often used in IoT ecosystems.

## Dataset Details

The model is trained on the **Creative Senz3D** dataset, a public resource provided by the Multimedia Technology and Telecommunications Laboratory at the University of Padova. 

* **Source:** [Creative Senz3D Dataset](https://lttm.dei.unipd.it/downloads/gesture/)
* **Scope:** Approximately 1,060 images featuring 11 distinct types of hand movements.
* **Diversity:** Gestures are performed by different individuals under varying lighting conditions to ensure model robustness.

### Class Separation
For the purpose of simple, binary IoT device toggling, the dataset has been manually grouped into two primary classes:

* **`on` Class:** Hand shapes displaying multiple extended fingers (e.g., open palms, "OK" signs).
* **`off` Class:** Hand shapes dominated by closed or clenched positions (e.g., fists, two-finger peace signs).

## Features

* **Real-Time Detection:** Processes camera feeds on the fly to detect user inputs.
* **Customizable Tasks:** Users can easily map the recognized `on` and `off` gestures to specific IoT actions, such as toggling smart plugs, adjusting lighting, or triggering automated routines.
* **Efficient Processing:** Lightweight enough to run without requiring a high-end GPU.
