---
layout: page
title: YOLO Object Detection
description: A responsive demo of YOLO object detection with gallery and interactive app.
img: assets/img/1.jpg
importance: 1
category: work
related_publications: false
---

## Project Overview

This project demonstrates the capabilities of **YOLO (You Only Look Once)**, a state-of-the-art real-time object detection system. Using deep convolutional neural networks, YOLO can detect multiple objects in an image with high speed and accuracy.

In this demo, we verify the integration of:
1.  **Photo Gallery**: Displaying detection results.
2.  **Video Embedding**: Hosting externally and embedding here.
3.  **Interactive App**: Using Hugging Face Spaces (Gradio) to run the model.

---

## Detection Gallery

Here are some examples of object detection results using the YOLO model.

<!-- Using raw HTML with no indentation to avoid markdown code block issues -->
<div class="row">
<div class="col-sm mt-3 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/1.jpg" title="Street View" class="img-fluid rounded z-depth-1" %}
</div>
<div class="col-sm mt-3 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/3.jpg" title="Traffic" class="img-fluid rounded z-depth-1" %}
</div>
<div class="col-sm mt-3 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/5.jpg" title="Pedestrians" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
    Left: Urban street scene. Middle: Highway traffic. Right: Pedestrian crossing. (Placeholder results)
</div>

---

## Video Demo

Below is a demonstration of real-time detection on video. 

<div class="row justify-content-center">
<div class="col-12 text-center">
<!-- 
    IMPORTANT: You cannot use the full URL (e.g. youtube.com/watch?v=ID).
    You MUST use the embed URL format: https://www.youtube.com/embed/YOUR_VIDEO_ID
    Example: For https://www.youtube.com/watch?v=Q404_255w2M, the ID is Q404_255w2M
-->
<iframe width="100%" height="500" src="https://www.youtube.com/embed/DvdgpXebZ_I" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
</div>
<div class="caption">
    YOLOv8 Real-time Detection Demo (Source: Ultralytics)
</div>

---

## Interactive Demo

Try the model yourself! This interactive demo is hosted on Hugging Face Spaces.

<div class="row justify-content-center">
<!-- 
    1. Create a Space on Hugging Face (https://huggingface.co/spaces)
    2. Choose 'Gradio' SDK.
    3. Copy the 'Embed this space' URL (found in the top right menu of your Space).
    4. Paste it below as the src.
-->
<iframe src="https://cekxm-yolo-demo.hf.space" frameborder="0" width="100%" height="900"></iframe>
</div>
