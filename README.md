## Deep-CNN-Image-Classifier

### Goal
The goal of this repo is to make use of CNN to predict images by AI. 

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

![Max Pooling](https://github.com/chingjie98/Deep-CNN-Image-Classifier/assets/35895182/e9ff2476-ff6a-4b4f-8e67-80a9b66cf0e9)





