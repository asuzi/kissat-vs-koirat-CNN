# Convolutional Neural Network (CNN) using pytorch
Deep learning using PyTorch

The code is made by me using a paper from Sentdex at https://pythonprogramming.net/convolutional-neural-networks-deep-learning-neural-network-pytorch/

The database used in cats vs dogs CNN has 24946 images of cats and dogs, for training I used 22452 images and testing 2494 images. (all images are 50px by 50px and grayscaled)

Output:

    Cat == 0 & Dog == 1
    Prediction :  tensor(0)
    Reality :  tensor(1)

    (...)

    Cat == 0 & Dog == 1
    Prediction :  tensor(0)
    Reality :  tensor(0)

    Total guesses:  2494
    Correct guesses:  1772
    Accuracy of the guesses:  0.711


