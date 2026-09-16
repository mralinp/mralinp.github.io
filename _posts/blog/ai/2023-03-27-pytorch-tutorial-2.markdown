---
layout: post
title:  "PyTorch Tutorial, Part 2: Datasets and DataLoaders"
author: "Ali Naderi"
img: "/assets/images/posts/blog/pytorch-tutorial/part-2/title.png"
date:   2023-03-27 17:15:23 +0330
categories:  blog ai machine-learning deep-learning python pytorch
brief: "PyTorch's Dataset and DataLoader classes, built up from a custom dataset to a full training loop, classifying real wine samples with a small neural network."
---
[Part 1](/blog/ai/machine-learning/deep-learning/python/pytorch/2023/03/22/pytorch-tutorial-1.html) trained on data that lived entirely in a few lines of Python — five numbers, or a scikit-learn dataset that loads fully into memory in one call. Real datasets are rarely that convenient: they're too large to hold in memory at once, they need shuffling and batching so the model doesn't just memorize the order they're stored in, and loading and preprocessing them shouldn't block the GPU from training while it waits. PyTorch's `Dataset` and `DataLoader` classes exist specifically to solve that, by cleanly separating two concerns that are easy to tangle together: *what* your data is and how to fetch one example of it (`Dataset`), and *how* to turn a stream of individual examples into shuffled, batched tensors ready for a training loop (`DataLoader`). This post builds both up from scratch, then — unlike a lot of tutorials that stop at "here's how to load data" — actually trains a real classifier with what we build, tying it back to Part 1's training loop.

# 1. A custom Dataset

We'll use the classic [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine) from the UCI Machine Learning Repository [1]: 178 wine samples, each with 13 chemical measurements (alcohol content, malic acid, ash, and so on) and a label — which of three cultivars the wine came from. It's a genuinely nice dataset to learn on: small enough to inspect by eye, real enough to have actual measurement noise, and a multi-class (not just binary) classification problem, which we didn't cover in Part 1. Download `wine.data` from the repository page and place it somewhere like `../data/wine/wine.data` relative to your script (I keep a `data/` directory alongside my code for exactly this).

Every PyTorch `Dataset` is a class inheriting from `torch.utils.data.Dataset` that implements exactly two methods: `__getitem__(self, index)`, returning the sample at that index, and `__len__(self)`, returning the total number of samples. That's the entire contract — PyTorch doesn't care *how* you fetch a sample (from a NumPy array already in memory, from a file you open lazily on each call, from a network request), only that these two methods work.

```python
import torch
import numpy as np

PATH_TO_DATASET = '../data/wine/wine.data'

class WineDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        # The file is one row per sample, comma-separated, label in column 0
        # and the 13 features in the remaining columns.
        raw_data = np.loadtxt(PATH_TO_DATASET, delimiter=',', dtype=np.float32)
        self.x = torch.from_numpy(raw_data[:, 1:])
        self.y = torch.from_numpy(raw_data[:, [0]])
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
```

Now we can use it like any indexable, sized Python object:

```python
dataset = WineDataset()
print(f"{len(dataset)} samples, {dataset.n_features} features each")
x_sample, y_sample = dataset[0]
print(f"x: {x_sample}, y: {y_sample}")
```

```txt
178 samples, 13 features each
x: tensor([1.4230e+01, 1.7100e+00, 2.4300e+00, 1.5600e+01, 1.2700e+02, 2.8000e+00,
        3.0600e+00, 2.8000e-01, 2.2900e+00, 5.6400e+00, 1.0400e+00, 3.9200e+00,
        1.0650e+03]), y: tensor([1.])
```

**If you'd rather skip the manual download** while learning, scikit-learn ships this exact dataset built in — `sklearn.datasets.load_wine()` returns the same 178-sample, 13-feature data without touching the filesystem. Either source is fine for this post; we'll stick with the custom-file version above because loading your *own* CSV or data file, not a bundled toy dataset, is the far more common real-world need this section is actually teaching.

That's it — it looks almost too simple, and that's the point. The class does nothing exotic; it just gives PyTorch a uniform interface so everything downstream (batching, shuffling, parallel loading) can be written once, generically, for *any* dataset that implements these two methods.

# 2. DataLoader

A `Dataset` alone only gets you one sample at a time. `torch.utils.data.DataLoader` wraps a `Dataset` and handles everything you actually want during training: grouping samples into batches, shuffling their order every epoch, and optionally loading them in parallel.

```python
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=2)
```

Each argument is doing a specific, named job worth understanding rather than copy-pasting:

- **`batch_size`** — how many samples to group into one tensor per training step. Training on one sample at a time is slow (poor use of GPU parallelism) and noisy (each update is based on very little information); training on the *entire* dataset at once (`batch_size = len(dataset)`) is the other extreme — one very stable but very infrequent and memory-hungry update per epoch. Something in between — commonly 16, 32, 64, ... — is the usual sweet spot, and it's a hyperparameter worth experimenting with rather than treating as fixed.
- **`shuffle=True`** — reshuffles the dataset's order at the start of every epoch. Without this, a model can pick up on spurious patterns tied to *storage order* (e.g., if all of one class happens to be grouped at the end of the file) rather than the actual features — always shuffle your training data unless you have a specific reason not to (sequential data like time series is the usual exception).
- **`num_workers`** — how many separate OS processes load and preprocess batches in the background, in parallel with the GPU training on the *previous* batch, so the GPU spends less time idle waiting for data. `0` (the default) loads data in the main process, which is simplest and fine for small in-memory datasets like this one; it starts mattering once loading involves real work (decoding images, reading from disk). One Windows/macOS-specific gotcha: with `num_workers > 0`, your training script needs its DataLoader-using code inside an `if __name__ == '__main__':` guard, because those platforms re-import your script in each worker process, and without the guard you get infinite recursive process spawning.

You can iterate a `DataLoader` directly, and it hands you one batch at a time:

```python
for batch_x, batch_y in dataloader:
    print(batch_x.shape, batch_y.shape)   # torch.Size([16, 13]) torch.Size([16, 1])
    break  # just peek at the first batch
```

Or, if you want a single batch without a loop (useful for quick inspection in a notebook), wrap it in `iter()` and call `next()`:

```python
dataiter = iter(dataloader)
x, y = next(dataiter)
print(x, y)
```

```txt
tensor([[1.3740e+01, 1.6700e+00, 2.2500e+00,  ..., 9.2000e-01, 3.2000e+00, 1.0600e+03],
        [1.1660e+01, 1.8800e+00, 1.9200e+00,  ..., 1.2300e+00, 2.1400e+00, 4.2800e+02],
        ...
        [1.3500e+01, 1.8100e+00, 2.6100e+00,  ..., ...              ]])
tensor([[1.], [2.], [1.], ...])
```

In real training loops you'll almost always use the `for batch_x, batch_y in dataloader:` form directly (we'll do exactly that in Section 4) — `iter()`/`next()` is mainly useful for debugging or peeking at a batch's shape and content interactively.

## 2.1 Splitting into train and test sets

`WineDataset` above loads the whole file as one dataset, but we need separate train and test splits, the same way `train_test_split` did in Part 1. `torch.utils.data.random_split` does the same job directly on a `Dataset` object, without pulling everything into NumPy first:

```python
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
```

Shuffling the test loader isn't necessary (we're not training on it, and order doesn't affect evaluation), so it's conventional to leave `shuffle=False` there — it's one less source of non-determinism when you're comparing runs.

# 3. Transforms

Sometimes you need to modify raw data before using it — normalizing values to a consistent scale (Part 1's `StandardScaler` did this for the breast cancer features), converting types, or, for images specifically, data augmentation: randomly cropping, flipping, or color-jittering each image slightly differently on every epoch, so the model sees a slightly different version of the "same" example each time and generalizes better instead of memorizing exact pixels. We can support this in our own `Dataset` by accepting an optional list of transform functions in the constructor, and applying them in order inside `__getitem__`:

```python
class WineDataset(torch.utils.data.Dataset):

    def __init__(self, transforms=None) -> None:
        raw_data = np.loadtxt(PATH_TO_DATASET, delimiter=',', dtype=np.float32)
        self.x = raw_data[:, 1:]
        self.y = raw_data[:, [0]]
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]
        self.transforms = transforms or []

    def __getitem__(self, index):
        sample = (self.x[index], self.y[index])
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __len__(self):
        return self.n_samples
```

A transform can be any callable, but the idiomatic PyTorch pattern is a class implementing `__call__`, so a transform can carry its own configuration (like the scale factor below) as ordinary attributes:

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

Pass a list of transforms to the constructor, and they're applied in order, each one's output feeding the next:

```python
dataset = WineDataset(transforms=[ToTensorTransformer(), MultiplierTransformer(10)])
x_sample, y_sample = dataset[0]
print(f"x: {x_sample}, y: {y_sample}")
```

```txt
x: tensor([1.4230e+02, 1.7100e+01, 2.4300e+01, 1.5600e+02, 1.2700e+03, 2.8000e+01,
        3.0600e+01, 2.8000e+00, 2.2900e+01, 5.6400e+01, 1.0400e+01, 3.9200e+01,
        1.0650e+04]), y: tensor([1.])
```

This hand-rolled pattern is worth building once to understand it, but for real work — especially with images — reach for `torchvision.transforms.Compose` instead of writing your own list-and-loop logic: it's the same idea (a sequence of callables, applied in order) with a large library of ready-made, well-tested transforms (resizing, cropping, normalization, augmentation) already implemented [2].

# 4. Putting it together: training a real classifier

Everything so far has been plumbing. Let's use it: a small neural network, trained on `WineDataset` through a `DataLoader`, classifying wine samples into one of three cultivars — combining this post's data pipeline with Part 1's training loop.

Two wrinkles specific to multi-class classification, worth flagging before the code: the Wine dataset's labels are `1`, `2`, `3`, but PyTorch's multi-class loss function expects `0`-indexed integer class labels, so we subtract 1; and instead of Part 1's single output neuron plus sigmoid (built for *binary* classification — one probability), we now need **one output per class**, interpreted as that class's un-normalized score, and PyTorch's `CrossEntropyLoss` — which combines a softmax (turning raw scores into a probability distribution over classes) and negative-log-likelihood loss into one numerically stable operation — expects exactly that: raw scores in, integer class index out, no manual softmax needed.

```python
import torch
import numpy as np

class WineDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        raw_data = np.loadtxt(PATH_TO_DATASET, delimiter=',', dtype=np.float32)
        self.x = torch.from_numpy(raw_data[:, 1:])
        # Labels are 1/2/3 in the file; CrossEntropyLoss wants 0-indexed classes.
        self.y = torch.from_numpy(raw_data[:, 0] - 1).long()
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)


class WineClassifier(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        # One hidden layer with a ReLU activation: without a nonlinearity between
        # layers, stacking two Linear layers would collapse into one big Linear
        # layer mathematically, gaining nothing over Part 1's single neuron.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)


model = WineClassifier(dataset.n_features, n_classes=3)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 5 == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                predicted_class = model(batch_x).argmax(dim=1)
                correct += (predicted_class == batch_y).sum().item()
                total += batch_y.size(0)
        print(f"epoch: {epoch}, loss: {loss.item():.3f}, test accuracy: {correct/total:.3f}")
```

```output
epoch: 0, loss: 0.812, test accuracy: 0.639
epoch: 5, loss: 0.213, test accuracy: 0.917
epoch: 10, loss: 0.084, test accuracy: 0.972
epoch: 15, loss: 0.031, test accuracy: 0.972
epoch: 20, loss: 0.014, test accuracy: 1.000
epoch: 25, loss: 0.028, test accuracy: 1.000
epoch: 30, loss: 0.006, test accuracy: 1.000
epoch: 35, loss: 0.040, test accuracy: 1.000
epoch: 40, loss: 0.045, test accuracy: 1.000
epoch: 45, loss: 0.003, test accuracy: 1.000
```

(Exact numbers will vary run to run — weight initialization and the train/test split from `random_split` are both randomized — but convergence to high accuracy within a few dozen epochs is expected on this dataset.)

Two small but easy-to-miss details in that loop: `model.train()` and `model.eval()` don't do any computation themselves — they just flip an internal flag that layers like dropout and batch normalization check to behave differently at train vs. test time (this network doesn't use either, so it's a no-op here, but it's the kind of habit worth building now rather than debugging silently-wrong eval-mode behavior later). And `.argmax(dim=1)` is how you go from `CrossEntropyLoss`'s raw per-class scores back to an actual predicted class: the index of the largest score, per sample.

This is the first genuinely complete pipeline in this series — data on disk, to a `Dataset`, to shuffled batches via a `DataLoader`, through a real (if small) multi-layer network, trained and evaluated with a proper train/test split. Every larger model you build afterward is this same shape, with a bigger network in the middle.

# 5. Conclusion

We built a custom `Dataset` around a real file on disk, wrapped it in a `DataLoader` for batching and shuffling, added support for transforms, and — tying it back to Part 1 — trained an actual small neural network end to end on that pipeline, with a proper train/test split and multi-class classification. That combination (`Dataset` + `DataLoader` + an `nn.Module` + a loss + an optimizer) is the complete skeleton of virtually every PyTorch project you'll build from here on; later posts in this series put bigger, more specialized pieces into that same skeleton rather than changing its shape.

# References

1. S. Aeberhard, M. Forina. Wine [Dataset]. UCI Machine Learning Repository, 1992. [doi.org/10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)
2. PyTorch. `torchvision.transforms` documentation. [pytorch.org/vision/stable/transforms.html](https://pytorch.org/vision/stable/transforms.html)
3. PyTorch. `torch.utils.data` documentation (Dataset, DataLoader, random_split). [pytorch.org/docs/stable/data.html](https://pytorch.org/docs/stable/data.html)
4. PyTorch. Datasets & DataLoaders tutorial. [pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
