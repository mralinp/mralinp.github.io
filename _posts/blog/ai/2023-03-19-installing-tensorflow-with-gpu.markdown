---
layout: post
title:  "Installing Tensorflow with GPU Support"
author: "Ali N. Parizi"
img: "/assets/images/posts/blog/installing-tensorflow-gpu/title.webp"
date:   2023-03-19 12:19:43 +0330
categories:  blog ai machine-learning deep-learning
brief: "The ultimate guide to installing the latest version of TensorFlow on Ubuntu 22.04 with GPU support."
---

# 1. Introduction

Deep learning's rise to prominence over the past decade has been remarkable. It has come to dominate nearly every major competition, driven several new lines of research, and given rise to new training methods. One of the most popular ways to handle deep learning models and solve complex computational problems is with the help of a deep learning framework.

One of the most popular such libraries is TensorFlow, widely regarded as one of the best tools for tackling almost any problem related to neural networks and deep learning. While it performs well on a CPU for smaller, simpler datasets, its real power comes from running on a Graphics Processing Unit (GPU).

Running on a GPU takes this framework's performance to another level entirely. However, one of the most frustrating parts of working with GPUs in deep learning is dealing with CUDA errors — a headache that most developers, researchers, and enthusiasts run into sooner or later.

In this article, we'll walk through how to install the latest version of TensorFlow with full GPU support.

We'll use Anaconda, since it's one of the best Python environments for machine learning work. To get started, let's install Anaconda on your computer — you can skip this step if you already have it installed on your Ubuntu machine.

<p align="center">
    <img class="img-light-bg" src="/assets/images/posts/blog/installing-tensorflow-gpu/keras-logo.png" width="40%"/>
</p>

# 2. Anaconda
Anaconda is a distribution of the Python and R programming languages for scientific computing (data science, machine learning, large-scale data processing, predictive analytics, and so on) that aims to simplify package management and deployment. The distribution includes data-science packages for Windows, Linux, and macOS, and is developed and maintained by Anaconda, Inc., founded by Peter Wang and Travis Oliphant in 2012. As an Anaconda, Inc. product it's also known as Anaconda Distribution or Anaconda Individual Edition, while the company's paid offerings are Anaconda Team Edition and Anaconda Enterprise Edition. For me, and probably for you and the vast majority of people, the free version does the job just fine.

On Debian-based distros (such as Ubuntu), start by installing a few required system libraries:

```console
$ sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
To install Anaconda, visit its official website [anaconda.com](https://www.anaconda.com/products/distribution) and download the latest installer, or run the command below:

```console
$ curl https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh | /bin/bash
```

Then follow the installer's prompts to complete the installation. Close and reopen your terminal window for the installation to take effect, or run `source ~/.bashrc` (or `~/.zshrc` if you're using zsh) to refresh the current one.

> **Note**: The installer will ask whether to initialize Anaconda Distribution by running `conda init`. Anaconda recommends answering "yes" — if you answer "no", conda won't modify your shell scripts at all. To initialize later, first run `source [PATH TO CONDA]/bin/activate` and then run `conda init`.

## 2.1 Creating a Conda Environment

Create a new conda environment named `tf` with the following command:
```console
$ conda create --name tf python=3.9
```
You can deactivate and activate it with the following commands:
```console
$ conda deactivate
$ conda activate tf
```

> **Note**: After installing Anaconda, the default conda environment activates automatically whenever you open a new terminal. I personally prefer not to activate it automatically — you can turn this off by running `$ conda config --set auto_activate_base False`.

# 3. Nvidia Driver, CUDA, and cuDNN
You'll need a proper Nvidia driver installed on your machine. If you haven't installed one yet, use the command below:

```console
$ sudo apt install nvidia-driver-515
```

To confirm it's installed properly, run the command below:

```console
$ nvidia-smi
```

```output
Mon Mar 19 12:19:49 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.86.01    Driver Version: 515.86.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:65:00.0  On |                  N/A |
|  0%   47C    P8    44W / 340W |   1325MiB / 10240MiB |      4%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1876      G   /usr/lib/xorg/Xorg                940MiB |
|    0   N/A  N/A      2034      G   /usr/bin/gnome-shell               48MiB |
|    0   N/A  N/A      3396      G   ...1/usr/lib/firefox/firefox      161MiB |
|    0   N/A  N/A      4658      G   ...816051303568945556,131072       42MiB |
|    0   N/A  N/A      4797      G   ...RendererForSitePerProcess      130MiB |
+-----------------------------------------------------------------------------+
```

Now install CUDA and cuDNN:

```console
(tf) $ conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
```
Next, configure the system paths. You'll need to run the following command every time you start a new terminal, after activating your conda environment:

```console
(tf) $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

For convenience, it's recommended that you automate this instead, so the paths are configured automatically whenever you activate the environment:

```console
(tf) $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
(tf) $ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

On Ubuntu 22.04, you'll also need to install NVCC:

```console
# Install NVCC
(tf) $ conda install -c nvidia cuda-nvcc=11.3.58
# Configure the XLA cuda directory
(tf) $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
(tf) $ printf 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/\nexport XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
(tf) $ source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Copy libdevice file to the required path
(tf) $ mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
(tf) $ cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
```

# 4. Installing TensorFlow
TensorFlow requires a recent version of pip, so upgrade pip first before installing TensorFlow itself:
```console
(tf) $ pip install --upgrade pip
(tf) $ pip install tensorflow
```
Now verify the GPU setup:
```console
(tf) $ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If a list of GPU devices is returned, TensorFlow has been installed successfully with GPU support.

```output
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

# References
- [*Installing Anaconda (anaconda.com)*](https://docs.anaconda.com/anaconda/install/index.html)
- [*Install TensorFlow with pip (tensorflow.org)*](https://www.tensorflow.org/install/pip)

