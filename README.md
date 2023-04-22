This is a possible rewrite for a single .ipynb file:

# Unet-SegmentationFlowers

In this notebook, we will use the U-Net architecture to perform semantic segmentation on the tf_flower dataset. Semantic segmentation is a computer vision task that involves assigning a label to each pixel in an image, indicating which object or class it belongs to. For example, in an image of a garden, semantic segmentation can identify which pixels belong to roses, sunflowers, daisies, etc.

The tf_flower dataset is a collection of images of various types of flowers, such as roses, sunflowers, daisies, tulips, and more. The dataset contains 3670 images with a resolution of 512x512 pixels. The dataset is available from TensorFlow Datasets (https://www.tensorflow.org/datasets/catalog/tf_flowers).

The U-Net model is a convolutional neural network that was originally designed for biomedical image segmentation. The model has an encoder-decoder structure, where the encoder gradually reduces the spatial resolution of the input image and extracts high-level features, while the decoder gradually increases the spatial resolution and produces the segmentation map. The model also uses skip connections between the encoder and decoder layers, which help preserve the low-level features and improve the segmentation accuracy. The U-Net model is described in this paper: https://arxiv.org/abs/1505.04597.

We will use TensorFlow and Keras to implement the U-Net model and train it on the tf_flower dataset. We will also show some examples of the segmentation results and how to use the model on our own images of flowers.

## Imports

First, let's import the necessary packages for this notebook:

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from config import *
from unet import *
```

## Data Loading and Visualization

Next, let's load the tf_flower dataset using TensorFlow Datasets. We will also create the training and validation splits using a 80/20 ratio:

```python
dataset = tfds.load('tf_flowers', split='train', as_supervised=True)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.2 * dataset_size)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
```

Let's also define some helper functions to preprocess the images and masks for our model:

```python
def resize_image(image, label):
  image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
  image = tf.cast(image, tf.float32) / 255.0
  label = tf.image.resize(label, [IMG_HEIGHT, IMG_WIDTH])
  label = tf.cast(label, tf.int32)
  return image, label

def one_hot_encode(image, label):
  label = tf.one_hot(label, NUM_CLASSES)
  return image, label

def augment_image(image, label):
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_brightness(image, 0.1)
  return image, label
```

We will resize the images and masks to 256x256 pixels, normalize the pixel values to [0,1], one-hot encode the labels to NUM_CLASSES (5 in this case), and apply some random augmentations to the training images:

```python
train_dataset = train_dataset.map(resize_image).map(one_hot_encode).map(augment_image).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_dataset = val_dataset.map(resize_image).map(one_hot_encode).batch(BATCH_SIZE)
```

Let's also define a function to display some images and masks from our dataset:

```python
def display(display_list):
  plt.figure(figsize=(15, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
```

Let's take a look at some examples from our training dataset:

```python
for image, mask in train_dataset.take(3):
  sample_image, sample_mask = image[0], mask[0]
  display([sample_image, sample_mask])
```

![Training Image](images/train1.jpg)
![Training Mask](images/train2.jpg)

![Training Image](images/train3.jpg)
![Training Mask](images/train4.jpg)

![Training Image](images/train5.jpg)
![Training Mask](images/train6.jpg
