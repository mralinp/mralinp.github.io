---
layout: post
title:  "Installing Tensorflow with GPU Support"
author: "Ali N. Parizi"
img: "/assets/images/posts/installing-tensorflow-gpu/title.webp"
date:   2023-03-19 12:19:43 +0330
categories:  blog ai machine-learning deep-learning
brief: "The Ultimate Guide To Install The Latest Version Of TensorFlow on your Ubuntu 22.04 With GPU Support."
---

# 1. Intro

The rise to prominence of deep learning over the past decade is spectacular. From dominating in almost every single competition with its innovative and groundbreaking technologies, it has also led to several new types of research and training methods. One of the most popular ways to handle deep learning models to solve complex computational problems is with the help of deep frameworks.

One such popular deep learning library to build and construct models to find solutions to numerous tasks is TensorFlow. TensorFlow is regarded as one of the best libraries to solve almost any question related to neural networks and deep learning. While this library performs effectively with most smaller and simpler datasets to achieve tasks on a CPU, its true power lies in the utilization of the Graphics Processing Unit (GPU).

The GPU improvises the performance of this deep learning framework to reach new heights and peaks. However, one of the most annoying issues that deep learning programmers, developers, and enthusiasts face is the trouble of CUDA errors. This experience is rather frustrating for most individuals because it is a common occurrence while dealing with deep learning models.

In this article, we will explore how to get the latest version of TensorFlow and stay updated with modern technology. 

We will use Anacoda because it's almost the best python environment for machine-learning operations. To get started, let's install anacoda on your computer. You can skip this step if you already have installed Anacoda on your Ubuntu machine. 

# 2. Anaconda
Anaconda is a distribution of the Python and R programming languages for scientific computing (data science, machine learning applications, large-scale data processing, predictive analytics, etc.), that aims to simplify package management and deployment. The distribution includes data-science packages suitable for Windows, Linux, and macOS. It is developed and maintained by Anaconda, Inc., which was founded by Peter Wang and Travis Oliphant in 2012. As an Anaconda, Inc. product, it is also known as Anaconda Distribution or Anaconda Individual Edition, while other products from the company are Anaconda Team Edition and Anaconda Enterprise Edition, both of which are not free. For me and probably you and almost 90% of people, the free version is good and does the job well for us. Installing anaconda requires installing For Debian based distros (such as Ubuntu) run the command below:

```console
$ sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
For installing Anaconda you can visit its official website [anaconda.com](https://www.anaconda.com/products/distribution) and download the latest installer version or Run the command below:

```console
$ curl https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh | /bin/bash
```

Then follow the installation process to complete it. Close and re-open your terminal window for the installation to take effect, or enter the command source ~/.bashrc (or ~/.zshrc if you are using zsh) to refresh the terminal.

> **Note**: The installer prompts you to choose whether to initialize Anaconda Distribution by running `conda init`. Anaconda recommends entering “yes”. If you enter “no”, then conda will not modify your shell scripts at all. To initialize after the installation process is done, first run source [PATH TO CONDA]/bin/activate and then run `conda init`.

## 2.1 Creating Conda Environment

Create a new conda environment named tf with the following command.
```console
$ conda create --name tf python=3.9
```
You can deactivate and activate it with the following commands.
```console
$ conda deactivate
$ conda activate tf
```

> **Note**: After installing Anaconda, the default conda environment will automatically activated when you open a new terminal. I personally prefer not to activate the environment automatically. You can turn off this feature running ```$ conda config --set auto_activate_base False```.

# 4. Nvidia Driver, CUDA and cuDNN
It is required you to install a proper Nvidia driver on your machine. If you haven't installed the Nvidia driver on your machine use the command below to install the driver:

```console
$ sudo apt install nvidia-driver-515
```

To confirm that it is installed properly run the command bellow:

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

Then install CUDA and cuDNN:

```console
(tf) $ conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
```
Configure the system paths. You can do it with the following command every time you start a new terminal after activating your conda environment.

```console
(tf) $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

For your convenience it is recommended that you automate it with the following commands. The system paths will be automatically configured when you activate this conda environment.

```console
(tf) $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
(tf) $ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

In Ubuntu 22.04, we have to install NVCC as well:

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

# 5. Installing Tensorflow
TensorFlow requires a recent version of pip, so upgrade your pip installation to be sure you're running the latest version.
```console
(tf) $ pip install --upgrade pip
(tf) $ pip install tensorflow
```
Verify the GPU setup:
```console
(tf) $ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
````
If a list of GPU devices is returned, you've installed TensorFlow successfully.

```output
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

# References
- [*Installing Anaconda (anaconda.com)*](https://docs.anaconda.com/anaconda/install/index.html)
- [*Install TensorFlow with pip (tensorflow.org)*](https://www.tensorflow.org/install/pip)

