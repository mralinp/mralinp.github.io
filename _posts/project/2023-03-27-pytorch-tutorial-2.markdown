---
layout: post
title:  "PyTorch Tutorial, Part 2: Datasets and Dataloaders"
author: "Ali N. Parizi"
img: "/assets/images/posts/projects/pytorch-tutorial/part-2/title.png"
date:   2023-03-27 17:15:23 +0330
categories:  project ai machine-learning deep-learning python
brief: "In this part we will learn how to make custom Datasets and use DataLoader in PyTorch."
---

# 1. Intro

Almost every machine learning algorithm and model works with data. Creating a Dataset and managing it with a Dataloader keeps your data manageable and helps simplify your machine learning pipeline. A Dataset stores all your data, and a Dataloader can be used to iterate through the data, manage batches, transform the data, and much more. Let's begin with a simple example. Assuming the popular Wine dataset as our target dataset, we want to load and use this dataset in PyTorch. Before getting started, we have to download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine). I usually place my data inside a data directory, so the path to this dataset will be `../data/wine/wine.data`.

In PyTorch we have a Dataset class which every dataset should inherit from. For a dataset we have to implement at least two methods, `__getitem__` and `__len__`, which return the item at the given index and the length of the dataset, respectively.

```python

import torch

PATH_TO_DATASET = '../data/wine/wine.data'

class WineDataset(torch.utils.data.Dataset):
    # default constructor
    def __init__(self) -> None:
        raw_data = np.loadtxt(PATH_TO_DATASET, delimiter=',', dtype=np.float32)
        self.x = torch.from_numpy(raw_data[:, 1:])
        self.y = torch.from_numpy(raw_data[:, [0]])
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]
    
    # returns item index of dataset (x, y)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    # returns the length of dataset
    def __len__(self):
        return len(self.x)
```

Now we are able to use this dataset as follows:

```python
dataset = WineDataset()
x_sample, y_sample = dataset[0]
print (f"x:{ x_sample}, y: {y_sample}")
```

```txt
x:tensor([1.4230e+01, 1.7100e+00, 2.4300e+00, 1.5600e+01, 1.2700e+02, 2.8000e+00,
        3.0600e+00, 2.8000e-01, 2.2900e+00, 5.6400e+00, 1.0400e+00, 3.9200e+00,
        1.0650e+03]), y: tensor([1.])
```

That's it, it looks easy, doesn't it?

# 2. Dataloader
To manage the dataset we can use PyTorch's Dataloader to prepare it for the training process. For example, it's responsible for creating batches and shuffling them during training.

```python
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=2)
dataiter = iter(dataloader)
data = next(dataiter)
x, y = data
print (x, y)
```

```txt
Output exceeds the size limit. Open the full output data in a text editor
tensor([[1.3740e+01, 1.6700e+00, 2.2500e+00, 1.6400e+01, 1.1800e+02, 2.6000e+00,
         2.9000e+00, 2.1000e-01, 1.6200e+00, 5.8500e+00, 9.2000e-01, 3.2000e+00,
         1.0600e+03],
        [1.1660e+01, 1.8800e+00, 1.9200e+00, 1.6000e+01, 9.7000e+01, 1.6100e+00,
         1.5700e+00, 3.4000e-01, 1.1500e+00, 3.8000e+00, 1.2300e+00, 2.1400e+00,
         4.2800e+02],
        [1.4340e+01, 1.6800e+00, 2.7000e+00, 2.5000e+01, 9.8000e+01, 2.8000e+00,
         1.3100e+00, 5.3000e-01, 2.7000e+00, 1.3000e+01, 5.7000e-01, 1.9600e+00,
         6.6000e+02],
        [1.1610e+01, 1.3500e+00, 2.7000e+00, 2.0000e+01, 9.4000e+01, 2.7400e+00,
         2.9200e+00, 2.9000e-01, 2.4900e+00, 2.6500e+00, 9.6000e-01, 3.2600e+00,
         6.8000e+02],
        [1.4130e+01, 4.1000e+00, 2.7400e+00, 2.4500e+01, 9.6000e+01, 2.0500e+00,
         7.6000e-01, 5.6000e-01, 1.3500e+00, 9.2000e+00, 6.1000e-01, 1.6000e+00,
         5.6000e+02],
        [1.3050e+01, 5.8000e+00, 2.1300e+00, 2.1500e+01, 8.6000e+01, 2.6200e+00,
         2.6500e+00, 3.0000e-01, 2.0100e+00, 2.6000e+00, 7.3000e-01, 3.1000e+00,
         3.8000e+02],
        [1.3510e+01, 1.8000e+00, 2.6500e+00, 1.9000e+01, 1.1000e+02, 2.3500e+00,
         2.5300e+00, 2.9000e-01, 1.5400e+00, 4.2000e+00, 1.1000e+00, 2.8700e+00,
         1.0950e+03],
        [1.2600e+01, 2.4600e+00, 2.2000e+00, 1.8500e+01, 9.4000e+01, 1.6200e+00,
         6.6000e-01, 6.3000e-01, 9.4000e-01, 7.1000e+00, 7.3000e-01, 1.5800e+00,
         6.9500e+02],
        [1.3500e+01, 1.8100e+00, 2.6100e+00, 2.0000e+01, 9.6000e+01, 2.5300e+00,
...
        [2.],
        [1.],
        [2.],
        [1.]])
```

# 3. Transformers

Sometimes we need to make changes to the raw data before using it. For example, with the data augmentation technique, we make small changes to the original data before using it in each training epoch. This makes our model more robust and less prone to overfitting. We can modify our dataset class to support transformer functions by defining an optional input in the constructor (`__init__`). Then, whenever we want to read an item from the dataset, the transformer functions will automatically run on the data before the result is returned. This will be very useful in our implementation, and we'll see more examples of it in the next parts of this series.

Here is the modified dataset class:

```python
class WineDataset(torch.utils.data.Dataset):
    
    def __init__(self, transforms=[]) -> None:
        raw_data = np.loadtxt('../data/wine/wine.data', delimiter=',', dtype=np.float32)
        self.x = raw_data[:, 1:]
        self.y = raw_data[:, [0]]
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]
        self.transforms = transforms
    
    def __getitem__(self, index):
        sample = (self.x[index], self.y[index])
        for transform in self.transforms:
            sample = transform(sample)
        print (sample)
        return sample
    
    def __len__(self):
        return len(self.x)
```

We can define a transformer as a callable class. Here are some examples:

```python
class ToTensorTransformer:
    def __call__(self, sample):
        x, y = sample
        return torch.from_numpy(x), torch.from_numpy(y)

class MultiplierTransformer:
    
    def __init__(self, factor: float):
        self.factor = factor
        
    def __call__(self, sample):
        x, y = sample
        x = x * self.factor
        return x, y
```

To use these transformers, we can easily pass them through the constructor while creating the dataset object. They'll then be applied to the data one after another, in the order they appear in the array.

```python
dataset = WineDataset(transforms=[ToTensorTransformer(), MultiplierTransformer(10)])
x_sample, y_sample = dataset[0]
print (f"x:{ x_sample}, y: {y_sample}")
```

```txt
(tensor([1.4230e+02, 1.7100e+01, 2.4300e+01, 1.5600e+02, 1.2700e+03, 2.8000e+01,
        3.0600e+01, 2.8000e+00, 2.2900e+01, 5.6400e+01, 1.0400e+01, 3.9200e+01,
        1.0650e+04]), tensor([1.]))
x:tensor([1.4230e+02, 1.7100e+01, 2.4300e+01, 1.5600e+02, 1.2700e+03, 2.8000e+01,
        3.0600e+01, 2.8000e+00, 2.2900e+01, 5.6400e+01, 1.0400e+01, 3.9200e+01,
        1.0650e+04]), y: tensor([1.])
```
