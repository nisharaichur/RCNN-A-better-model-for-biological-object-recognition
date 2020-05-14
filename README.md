# Recurrent Convolutional Neural Networks: A Better Model of Biological Object Recognition
Implements the research paper: https://www.frontiersin.org/articles/10.3389/fpsyg.2017.01551/full 
Here I have used the Digit-debris dataset to compare the performaces of recurrent convolutional neural networks with bottom-up (B), lateral (L), and top-down (T) connections with that of the conventional CNN(feed-forward alone).

# Prerequisites
- tensorflow 2.0. 
- keras 
- numpy
- matplotlib

# Dataset:
4 types of data in Digit-debris dataset:
  - No Debris - 0 frangments(original images)
  - Light Debris - 10 fragments
  - Moderate Debris - 30 fragments
  - Heavy Debris - 50 fragments
  
<img src="RNN_Images/data_set_generation.PNG" height="300" >

# Error rates of each model 
- Percentage error increases as the number of number debris increases (BLT has the lowest error with the highest number of debris)
- Feeb-forward performs the worst with higest error rate (90%) for 50 debris
- Models with lateral connections perfomed convincingly better than feed-forward alone

| Models  | Feedforward | Lateral-BL(%) | Top-down-BT(%)| Lateral top down-BLT(%) |
| ------ | ------ | ------ | ------ | ----- | 
|No Debris | 0.649 | 0.709 | 0.919 | 0.668 | 
|Light Debris(10)| 19.50 | 2.969 | 14.11 | 4.549 | 
|Moderate Debris(30)| 91 | 12.650 | 90.5 | 21.697 |
|Heavy Debris(50) | 91 | 16.708 | 90.53 | 21.11|

# Files description
- generate_add_bebris.py: This file generates the debris and the images of desired number and adds the bedris onto the images.
- add_debris.py: This file adds the already generated debris onto the images
- main.py: This file imports the model and calculates the the training function, which calculates the gradient, loss, accuracy, activations.
- recurrent_lateral_BL.py: This file when imported in main.py crates a recurrent model with lateral connections alone(BL)
- recurrent_topDown_BLT.py: This file when imported in main.py crates a recurrent model with lateral connections and topdown connections(BLT).
- top_down_BT.py: This file when imported in main.py crates a recurrent model with topdown connections alone(BT).
- feedforward_BF.ipynb: This is the .ipynb file which has a keras implementation for a feedforward model
- Statistical_Test.ipynb: This is the .ipynb file which calculates the pair-wise McNemar test between any two models.








