# Recurrent Convolutional Neural Networks: A Better Model of Biological Object Recognition
Implements the research paper: https://www.frontiersin.org/articles/10.3389/fpsyg.2017.01551/full 
Here I have used the digit debris dataset to compare the performaces of recurrent convolutional neural networks with bottom-up (B), lateral (L), and top-down (T) connections with that of the conventional CNN(feed-forward alone).
# Prerequisites

tensorflow 2.0. keras, numpy, matplotlib

# Dataset:
4 types of dataset:
  - No Debris
  - Light Debris - 10 fragments
  - Moderate Debris - 30 fragments
  - Heavy Debris - 50 fragments
  

| Models  | Feedforward | Lateral-BL(%) | Top-down-BT(%)| Lateral top down-BLT(%) |
| ------ | ------ | ------ | ------ | ----- | 
|No Debris | 0.649 | 0.709 | 0.919 | 0.668 | 
|Light Debris(10)| 19.50 | 2.969 | 14.11 | 4.549 | 
|Moderate Debris(30)| 91 | 12.650 | 90.5 | 21.697 |
|Heavy Debris(50) | 91 | 16.708 | 90.53 | 21.11|



# generate_add_bebris.py
This file generates the debris and the images of desired number and adds the bedris onto the images.

# add_debris.py
This file adds the already generated debris onto the images


Train folder: contains the 100,000 images for each type of dataset

Validate folder: contains 10,000 images for each type of dataset

Test folder: contains 10,000 images for each type of dataset


# main.py:
This file imports the model and calculates the the training function, which calculates the gradient, loss, accuracy, activations.

# recurrent_lateral_BL.py:
This file when imported in main.py crates a recurrent model with lateral connections alone(BL)

# recurrent_topDown_BLT.py
This file when imported in main.py crates a recurrent model with lateral connections and topdown connections(BLT).

# top_down_BT.py
This file when imported in main.py crates a recurrent model with topdown connections alone(BT).

# feedforward_BF.ipynb
This is the .ipynb file which has a keras implementation for a feedforward model

# Statistical_Test.ipynb
THis is the .ipynb file which calculates the pair-wise McNemar test between any two models.








