---
layout: post
title: "What is an AutoEncoder?"
author: "Ali N. Parizi"
img: "/assets/images/posts/blog/auto-encoder/title.png"
date:   2022-09-16 12:21:13 +0330
categories: blog ai machine-learning deep-learning
brief: "One of the most popular deep architectures is the variety of AutoEncoders. This article is a straightforward walkthrough to get familiar with AutoEncoders."
---

# 1. Intro
An autoencoder is an unsupervised artificial neural network that learns to compress and encode data efficiently, then learns to reconstruct that data from the reduced, encoded representation into something as close to the original input as possible.
By design, an autoencoder reduces the dimensionality of data by learning to ignore the noise in it.
Here is an example of an input/output pair from the MNIST dataset passed through an autoencoder.

<p align="center">
  <img src="/assets/images/posts/blog/auto-encoder/ae-arch.jpeg" />
    <br>
    <span>a simple AutoEncoder</span>
</p>

## 1.1 AutoEncoder Components

An autoencoder consists of four main parts:

1. **Encoder**: learns how to reduce the input's dimensionality and compress it into an encoded representation.

2. **Bottleneck**: the layer that holds the compressed representation of the input data. This is the lowest-dimensional representation the network produces.

3. **Decoder**: learns how to reconstruct the data from the encoded representation, aiming to get as close to the original input as possible.

4. **Reconstruction Loss**: the metric that measures how well the decoder is performing, i.e. how close the output is to the original input.

Training then uses backpropagation to minimize the network's reconstruction loss. At this point you're probably wondering why you'd train a neural network just to reproduce its own input. This article walks through the most common use cases for autoencoders. Let's get started:


$$ Loss = \lVert X - \hat{X} \rVert_{2}^{2} $$


## 1.2 Problem Statement

The network architecture for an autoencoder can vary — a simple feedforward network, an LSTM, or a convolutional neural network — depending on the use case. This article uses a CNN to solve a simple problem: removing an intrusive piece of text from a picture. You've probably noticed that many photography websites and photographers stamp a signature or watermark onto their images, which keeps other people from lifting their photos, paintings, and other artwork without credit.

For example, here's a sample photo taken by my psychologist friend Reza Parizi ([reza__parizi](https://www.instagram.com/reza__parizi/)):

<p align="center">
  <img width="70%" src="/assets/images/posts/blog/auto-encoder/reza-parizi.jpg" />
</p>

We can treat this kind of text — commonly known as a [**watermark**](https://en.wikipedia.org/wiki/Watermark) — as static noise added to the picture, and look for a filter (or set of filters) that removes it. Denoising is one of the main use cases for autoencoders, so let's put one to work on this problem.

# 2. Preparing the Data

As with any deep learning model, the first thing we need is data. For this problem I used the popular [**Stanford Cars**](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset and stamped the static text "Hot-Tube" onto each image as a synthetic watermark. Say the dataset lives in a directory named `datasets/car`, with the training images inside a subdirectory called `train`. First, we import the required modules:

```python
import tensorflow as tf
import numpy as np
import keras
import keras.layers
import os
import matplotlib.pyplot as plt
import cv2
```

Next, we load the dataset:

```python
path_to_train_imgs = './datasets/cars/train'
train_imgs_list = os.listdir(path_to_train_imgs)
train_imgs_list = [f"{path_to_train_imgs}/{path}" for path in train_imgs_list]
```

These are the original, unmodified images, which we'll treat as ground truth. To generate the corresponding inputs, we have two options:

1. Load all the data into memory up front and loop over it, adding the watermark text to each image.
2. Write a `DataGenerator` that adds the watermark to each image on the fly, while assembling a batch.

The first approach isn't a great option, especially with images: image datasets tend to be large, and loading that much data into memory at once risks running out of memory and crashing the program.

A data generator instead loads only the slice of data needed for the current batch, which avoids that problem entirely. With that motivation out of the way, let's write one for our use case.

```python
class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_list, batch_size=16, shuffle=True):
        self.image_list = image_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def load_img(self, path: str) -> np:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256,256))
        return img
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __len__(self):
        return len(self.image_list)//self.batch_size
    
    def add_text(self, img: np) -> np:
        return cv2.putText(img=img, text='Hot-Tube', org=(46, 128), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        img_paths = [self.image_list[i] for i in indexes]
        
        Y = [self.load_img(img_path) for img_path in img_paths]
        X = [self.add_text(np.array(img)) for img in Y]
        
        return np.array(X)/255, np.array(Y)/255
```

Let's see it in action:

```python
def plt_img(img: np, title: str = None):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)

validation_imgs, train_imgs = train_imgs_list[:100], train_imgs_list[100:]
train_data_generator = DataGenerator(image_list=train_imgs)
validation_data_generator = DataGenerator(image_list=validation_imgs)

# Showing some sample images
X, Y = validation_data_generator[0]

plt.figure(figsize=(10,10))
cnt = 1;
for i in range(3):
    plt.subplot(3,2, cnt)
    plt_img(Y[i], "Original Image")
    cnt += 1
    plt.subplot(3, 2, cnt)
    plt_img(X[i], "Augmented Image")
    cnt += 1
plt.show()
```
<p align="center">
  <img width="70%" src="/assets/images/posts/blog/auto-encoder/sample_data.png" />
</p>

# 3. Building the Model

We'll train a convolutional autoencoder on the cars dataset. Here's a simple nine-layer network that does the job:

```python
inputs = keras.layers.Input((256,256,3))
x = keras.layers.Conv2D(128, 3, padding='same', activation="relu")(inputs)
x = keras.layers.Conv2D(64, 3, padding='same', activation="relu")(inputs)
x = keras.layers.Conv2D(32, 3, padding='same', activation="relu")(inputs)
x = keras.layers.MaxPool2D()(x) # 128x128
x = keras.layers.Conv2D(8, 3, padding='same', activation="relu", name="bottle-neck")(x) # 128*128
x = keras.layers.UpSampling2D()(x) # 256x256
x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
outputs = keras.layers.Conv2D(3, 1, padding='same', activation='relu')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])
model.summary()
```
```output
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 256, 256, 3)]     0         
                                                                 
 conv2d_2 (Conv2D)           (None, 256, 256, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 128, 128, 32)     0         
 )                                                               
                                                                 
 bottle-neck (Conv2D)        (None, 128, 128, 8)       2312      
                                                                 
 up_sampling2d (UpSampling2D  (None, 256, 256, 8)      0         
 )                                                               
                                                                 
 conv2d_3 (Conv2D)           (None, 256, 256, 32)      2336      
                                                                 
 conv2d_4 (Conv2D)           (None, 256, 256, 64)      18496     
                                                                 
 conv2d_5 (Conv2D)           (None, 256, 256, 128)     73856     
                                                                 
 conv2d_6 (Conv2D)           (None, 256, 256, 3)       387       
                                                                 
=================================================================
Total params: 98,283
Trainable params: 98,283
Non-trainable params: 0
_________________________________________________________________
```

I used mean squared error (MSE) as the loss, since the goal is simply for the model's output to match the original image as closely as possible. MSE does the job well here, though a loss that compares images region by region would likely work better for higher-resolution images or more general datasets. For this task, though, MSE is enough.

## 3.1 TensorBoard

To monitor training as it runs, we can use TensorBoard, a visualization tool built by the TensorFlow team. TensorBoard watches a directory (conventionally named `logs`) and renders the training logs it finds there — things like training and validation loss and accuracy — as live charts. It can also display images generated during training, such as the model's predictions at the end of each epoch, which is handy for judging by eye when the model has converged enough to stop training.

To capture those predictions, we define a custom callback that writes an image summary at the end of every epoch.

```python
class TensorBoardImageCallBack(keras.callbacks.Callback):
    def __init__(self, log_dir, image, noisy_image):
        super().__init__()
        self.log_dir = log_dir
        self.image = image
        self.noisy_image = noisy_image

    def set_model(self, model):
        self.model = model
        self.writer = tf.summary.create_file_writer(self.log_dir, filename_suffix='images')

    def on_train_begin(self, _):
        self.write_image(self.noisy_image, 'Noisy Image', 0)
        self.write_image(self.image, 'Original Image', 0)

    def on_train_end(self, _):
        self.writer.close()

    def write_image(self, image, tag, epoch):
        image_to_write = np.copy(image)
        with self.writer.as_default():
            tf.summary.image(tag, image_to_write, step=epoch)

    def on_epoch_end(self, epoch, logs={}):
        denoised_image = self.model.predict_on_batch(self.noisy_image)
        self.write_image(denoised_image, 'Denoised Image', epoch)
        
tensorboard_callback = TensorBoardImageCallBack('./logs', Y[1:2], X[1:2])
tensorboard_callback_loss = tf.keras.callbacks.TensorBoard(log_dir="./logs")

```

## 3.2 Training the Model

Now it's time to train the model. The code below is configured for 100 epochs, though I stopped training early, after 25, once the results looked good enough. Running the full 100 would likely push the model to a noticeably better convergence.

```python
history = model.fit(
            train_data_generator,
            epochs=100,
            validation_data=validation_data_generator,
            callbacks=[tensorboard_callback, tensorboard_callback_loss]
        )
```

```output
...
502/502 [==============================] - 86s 171ms/step - loss: 6.6057e-04 - accuracy: 0.8184 - val_loss: 5.3678e-04 - val_accuracy: 0.8163
```

As the output above shows, after roughly 25 epochs the validation reconstruction loss is down to 5.3678e-04, which is already solid, and would likely improve further with the full 100 epochs. If we now feed the trained model a new image from the test set, the reconstruction loss stays low. But if we feed it something quite different from what it was trained on — an outlier or anomaly — the reconstruction loss spikes, because the network simply doesn't know how to reproduce it. That behavior is itself another common use case for autoencoders: anomaly detection.

<p align="center">
  <img width="70%" src="/assets/images/posts/blog/auto-encoder/sample_prediction.png" />
</p>

One more thing worth noting: nothing stops you from using the encoder and decoder halves independently — the encoder alone to compress data, or the decoder alone to reconstruct it from a stored encoding. Here, we reduced the input image from $$256 \times 256 \times 3$$ down to $$128 \times 128 \times 8$$ at the bottleneck layer. Storing that compressed representation instead of the raw image shrinks the storage footprint by a factor of 1.5. Scaled up, a 900MB video would come down to about 600MB — a meaningful saving for storage-constrained applications.


$$\frac{256 \times 256 \times 3}{128 \times 128 \times 8} = 1.5$$

<p align="center">
  <img class="img-light-bg" src="/assets/images/posts/blog/auto-encoder/loss.png" />
  <br>
  <span>Model loss per epoch</span>
</p>

> Note: in the figure above, the red line is validation loss and the blue line is training loss, per epoch.

Beyond denoising and anomaly detection, autoencoders are also widely used for learning compact, lower-dimensional data representations, and for super-resolution — enhancing an image's quality beyond its original resolution. We won't cover super-resolution here, but it's a natural follow-up experiment for a future article.

In this article, we used an autoencoder to remove a static watermark from a set of images — one of many practical use cases for the architecture. I hope you enjoyed the walkthrough. Stay tuned for more!

# References

1. [*G. E. Hinton, & R. R. Salakhutdinov (2006). Reducing the Dimensionality of Data with Neural Networks. Science, 313(5786), 504-507.*](https://paperswithcode.com/method/autoencoder)


