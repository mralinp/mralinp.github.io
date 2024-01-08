---
layout: post
title:  "UNet: One of the most simple networks for segmentation"
author: "Ali N. Parizi"
img: "/assets/images/posts/blog/unet/brain.png"
date:   2023-08-20 23:02:23 +0330
categories: blog ai machine-learning deep-learning
brief: "In this article, we perform image classification on the MNIST dataset with custom implemented LeNet-5 neural network architecture."
---
# 1. Intro
The u-net is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. It has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin.

UNet is a convolutional neural network (CNN) architecture used for image segmentation tasks. It was introduced by researchers Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015. The name "UNet" comes from its U-shaped architecture, which consists of a contracting path (left side) and an expansive path (right side), connected by a central bottleneck.

Here's a brief overview of its structure and purpose:

- Contracting Path: The left side of the U-shaped architecture consists of repeated applications of convolutional layers and max-pooling operations. This part of the network is responsible for capturing the context and reducing the spatial dimensions of the input image.

- Bottleneck: At the bottom of the U, there is a bottleneck layer consisting of convolutional layers without max-pooling, which helps in capturing the most essential features from the contracted input.

- Expansive Path: The right side of the U-shaped architecture involves the use of transposed convolutions (also known as deconvolutions or up-sampling) to increase the spatial dimensions of the representation. The expansive path helps in generating a segmented output that has the same resolution as the input image.

- Skip Connections: One of the key innovations of UNet is the use of skip connections. During the expansive path, the high-resolution features from the contracting path are concatenated with the features at the corresponding level in the expansive path. These skip connections allow the network to preserve fine-grained details, which is crucial for accurate segmentation.

UNet is widely used in various applications, especially in medical image analysis, where precise segmentation of organs or anomalies is required. Its architecture and the use of skip connections make it effective in capturing both local details and global context, making it suitable for tasks where pixel-level accuracy is essential.

<p align="center">
    <img src="/assets/images/posts/blog/unet/arch.png">
    <br>
    <span>architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations.</span>
</p>

