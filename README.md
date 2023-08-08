## Deep-CNN-Image-Classifier

### Goal
The goal of this repo is to make use of convolutional neural network to predict images by AI. 

### Motivation
The motivation behind this repo is to get familiar with how CNN works with images before applying ANN to the flattened image layer. <br>
I have had previous experience with ANN, but not with CNN. Hence this repo will give me a better understanding and learning experience. 

### How CNN Works 

### 1. Feature Detection (Convolution Operation)
Get the important parts of a image, compress it into a feature map
<br>
Run the ReLu function to **non-linearize** the images. The purpose is most images are not linear with each other, since they have different angles, different colors, different properties so on. So if images are linearized with each other, they will be less accurate.

### 2. Pooling
Important so that however the image is positioned, the CNN can recognize. (To ensure **spatial invariance**) <br>
Doesn’t matter if it is looking left or right, so long as the feature (i.e., the long stripe along the cheetah's nose) is present, it will be recognized. <br>
Helps with reducing size also, just like feature detection above. And avoid over-fitting. And helps to preserve the main features. <br>

Here is the visual representation: <br>

![Max Pooling](https://github.com/chingjie98/Deep-CNN-Image-Classifier/assets/35895182/e9ff2476-ff6a-4b4f-8e67-80a9b66cf0e9) <br>

Max Pooling : we take the maximum within a certain frame on the left, then place the maximum on the corresponding pixel on the right. There are other ways as well such as Mean Pooling. 

### ** Pooling and Feature Detection can be done together. 

![First process](https://github.com/chingjie98/Deep-CNN-Image-Classifier/assets/35895182/d5ee2c54-7ffb-4a1b-bb20-ed57b902b40c) <br>

After convolution, the input image is called convolt image. <br>
After pooling, the convolt image is called pooled image.<br>

### 3. Flattening

![flattening](https://github.com/chingjie98/Deep-CNN-Image-Classifier/assets/35895182/1b89308d-1da8-4422-bab7-3ffb46972a84)

Convert the Pooled feature map, into a column to be used as an input layer of future ANN. 

![flattening2](https://github.com/chingjie98/Deep-CNN-Image-Classifier/assets/35895182/cb184a8c-60b3-4944-9698-297a0e45648a)

### 4. Overall process of CNN

Image is convoluted, pooled and flattened. Then the flattened feature map will be used as an input layer of ANN. 

Thereafter, let’s say our goal is to predict if the image is a Cat or Dog. Maybe 1 result say 80% Dog, but turns out to be cat. There will be back-propagation, as well as a Loss Function. In CNN, the loss function is known as cross-entrophy, which the CNN will automatically aim to reduce through back-propagation. 

During **back-propagation**, the **weights** are adjusted, as u can see which are the blue lines below. Another thing that is adjusted, is the **feature detectors**. Because what if we are looking at the wrong features? So both the weights and feature detectors undergo gradient descent. 

After back-propagation, forward-propagation occurs again. Then the whole cycle repeats.

![Process](https://github.com/chingjie98/Deep-CNN-Image-Classifier/assets/35895182/a0053a42-2e1d-4084-8d37-80ba676103c5)

What happens during the forward and backward propagation?

![Untitled](https://github.com/chingjie98/Deep-CNN-Image-Classifier/assets/35895182/63616069-25c6-4912-804b-95140795e578)

We can see that, the neuron with 0.9, 1 and 1 are firing up rapidly. It is signalling to both Dog and Cat. So for (i.e., maybe like floppy ears, big eyes and wet nose) the dog knows that it is resembling him, so it will adjust the weight to him accordingly. The cat will also roughly know that these are not his features, so it will be focusing on other neurons instead. 

The way that the dog knows that these neurons / features are for him, is because during back-propagation in the loss function, it is minimized, hence it is ‘rewarded’ with these features. 

The way that the output neuron knows that these neurons / features are for him, is because during back-propagation in the loss function, it is minimized, hence it is being ‘rewarded’ with these features when they guess correctly dog / cat.

![Untitled (1)](https://github.com/chingjie98/Deep-CNN-Image-Classifier/assets/35895182/321123de-bf9d-4b34-8b72-4950cf134afb)

![Untitled (2)](https://github.com/chingjie98/Deep-CNN-Image-Classifier/assets/35895182/3fa132be-971e-4b2f-8ed4-e3396fbdcf7e)
