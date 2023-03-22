---
layout: post
title:  "PyTorch Tutorial, Part 1: Installation and The basics"
author: "Ali N. Parizi"
img: "/assets/images/projects/pytorch-tutorial/title.png"
date:   2023-03-22 10:07:17 +0330
categories:  project ai machine-learning deep-learning python
brief: "PyTorch is the most popular deep-learning framework which is used by many researchers on the feild of machine learning and deep learning. I thing any body on this field shoud know this framework and use it on their implimentations."
---


# 1. Intro

PyTorch is a fully featured framework for building deep learning models, which is a type of machine learning that’s commonly used in applications like image recognition and language processing. Written in Python, it’s relatively easy for most machine learning developers to learn and use. PyTorch is distinctive for its excellent support for GPUs and its use of reverse-mode auto-differentiation, which enables computation graphs to be modified on the fly. This makes it a popular choice for fast experimentation and prototyping.

PyTorch is a machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing, originally developed by Meta AI and now part of the Linux Foundation umbrella. It is free and open-source software released under the modified BSD license.

PyTorch is the work of developers at Facebook AI Research and several other labs. The framework combines the efficient and flexible GPU-accelerated backend libraries from Torch with an intuitive Python frontend that focuses on rapid prototyping, readable code, and support for the widest possible variety of deep learning models. Pytorch lets developers use the familiar imperative programming approach, but still output to graphs.  It was released to open source in 2017, and its Python roots have made it a favorite with machine learning developers.

Significantly, PyTorch adopted a Chainer innovation called reverse-mode automatic differentiation. Essentially, it’s like a tape recorder that records completed operations and then replays backward to compute gradients. This makes PyTorch relatively simple to debug and well-adapted to certain applications such as dynamic neural networks. It’s popular for prototyping because every iteration can be different.

PyTorch is especially popular with Python developers because it’s written in Python and uses that language’s imperative, define-by-run eager execution mode in which operations are executed as they are called from Python. As the popularity of the Python programming language persists, a survey identified a growing focus on AI and machine learning tasks and, with them, greater adoption of related PyTorch. This makes PyTorch a good choice for Python developers who are new to deep learning, and a growing library of deep learning courses are based on PyTorch. The API has remained consistent from early releases, meaning that the code is relatively easy for experienced Python developers to understand.

PyTorch’s particular strength is in rapid prototyping and smaller projects. Its ease of use and flexibility also makes it a favorite for academic and research communities.

Facebook developers have been working hard to improve PyTorch’s productive applications. Recent releases have provided enhancements like support for Google’s TensorBoard visualization tool, and just-in-time compilation. It has also expanded support for ONNX (Open Neural Network Exchange), which enables developers to match with deep learning frameworks or runtimes that work best for their applications.

# 2. Installing PyTorch

You can folow this tutorial using some online platforms such as [Google Colab](https://colab.research.google.com) or [Kaggle](https://kaggle.com) which give you a python environment via a jupyter note book and a proper GPU to meet your needs during learning process and even doing small projects and homeworks. If you prefer using these platforms you can skip this section but if you want to use pythorch on your local machine and use your own GPU here is the installation steps that you should folow. 

## 2.1 Installing Nvidia driver
If you have a gpu on your machine (most of us use Nvidia GPSs) you have to install a proper driver on your system. This tutorial is based on Ubuntu 22.04 if you have Windows machine or other linux distros its on your own to find the alternatives of the following steps to complete the installation process. Install Nvidia Driver using:

```console
$ sudo apt install nvidia-driver-515
```

## 2.2 Installing PyTorch

You can folow the steps on PyTorch official website [pytorch.org](https://pytorch.org/get-started/locally/) to install it locally or stay with me. 

If you haven't installed Anaconda on your machine download and install anaconda then create a conda environment:

```console
$ conda create --name torch python=3.9
```

After creating the environment activate it using:

```console
$ conda activate torch
```

Then use pip to install PyTorch:

```console
$ pip install torch torchvision torchaudio
```

It will take some time but will install pytorch and all gpu requirements on your machine.

to test that if gpu is supported, open a python file and run the code below:

```python
import torch
print (f"Is GPU supported? {'Yes' if torch.cuda.is_avaiable() else 'No'}")
```

```output
Is GPU supported? Yes
```

Well done, you have installed PyTorch on your computer and ready to go through this tutorial.

# 3. Tensor basics
In pytorch every thing is based on tensor operations. Tensor looks like a multi dimentional array in python which contains numeric data on each of its elements.

```python
import torch
tensor = torch.empty(2,2) # this will create a 2*2 tensor with random values
print (tensor)
```

```output
 tensor([[0.12, 0.19], 
         [0.09, 0.82]])
```

## 3.1 Gradiant calculation

## 3.2 Optimizers 

## 3.3 Gradian Decent

# References
 