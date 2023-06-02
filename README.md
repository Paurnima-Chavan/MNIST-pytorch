
# Handwritten Digit Recognition using Convolutional Neural Networks with Pytorch

The MNIST dataset comprises 70,000 grayscale images of handwritten digits, each measuring 28 x 28 pixels and annotated by humans. It is a subset of a larger dataset made accessible by NIST, the National Institute of Standards and Technology. In this example, we will develop a handwritten digit recognizer utilizing a convolutional neural network (CNN) trained on the MNIST dataset, employing the PyTorch framework.


![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)


## Documentation

This documentation provides an overview of the PyTorch model used for classifying handwritten digits from the MNIST dataset. The model architecture consists of convolutional neural networks (CNNs) and fully connected layers, which are designed to achieve accurate digit recognition.


## Model Architecture

The model, implemented in the Net class, follows a sequential structure that defines the layers and operations performed on the input data.

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

### Model Summary

| Layer (type) | Output Shape | Param #  |
| ------------- |-------------:| -----:|
| Conv2d-1 | [-1, 32, 26, 26] | 320 |
| Conv2d-2 | [-1, 64, 24, 24] | 18, 496 |
| Conv2d-3 | [-1, 128, 10, 10] | 73, 856 |
| Conv2d-4 | [-1, 256, 8, 8] | 295, 168|
| Linear-5 | [-1, 50] | 204, 850 |
| Linear-6 | [-1, 10]  | 510 |

--------

Total params: 593,200

Trainable params: 593,200

Non-trainable params: 0

--------




### Convolutional Layers
- conv1: This layer applies a 2D convolution operation with 32 filters and a kernel size of 3 to the input.
- conv2: It performs another 2D convolution with 64 filters and a kernel size of 3 on the output of conv1.
- conv3: This layer applies a 2D convolution with 128 filters and a kernel size of 3 to the output of conv2.
- conv4: It performs a final 2D convolution with 256 filters and a kernel size of 3 on the output of conv3.


### Fully Connected Layers
- fc1: This layer is a fully connected layer with 4096 input features and 50 output features.
- fc2: It is the final fully connected layer with 50 input features and 10 output features, representing the 10 possible digit classes.

## Forward Pass
The forward method defines the forward pass of the model, where the input x undergoes a series of operations to generate the predicted digit class probabilities.

- x is passed through conv1, followed by a ReLU activation function.
- The result is then processed by conv2, followed by a ReLU activation function and max pooling operation.
- The output is further passed through conv3, followed by a ReLU activation function.
- conv4 is applied to the result, followed by a ReLU activation function and max pooling operation.
- The output is reshaped using the view function to match the input size of the fully connected layers.
- The reshaped output is passed through fc1, followed by a ReLU activation function.
- Finally, the output is processed by fc2 to obtain the logits, which are transformed into a log-softmax probability distribution using the log_softmax function.

## Usage

To use the model for handwritten digit recognition, followed below steps:

- Instantiate an instance of the Net class.
- Load the MNIST dataset and preprocess it as required.
- Pass the preprocessed input through the model's forward method to obtain the predicted digit class probabilities.
- We have used SGD (Stochastic Gradient Descent) optimizer, which will be used to update the model's parameters during training. The learning rate is set to 0.01, and the momentum is set to 0.9
- Trained the model for the specified number of epochs, printing the epoch number, and performs training and testing steps in each epoch. The learning rate is adjusted using the scheduler, allowing better optimization over time. The train and test functions are responsible for the actual training and testing processes, respectively.
- Finally, plotted the training and testing accuracy as well as the training and testing loss. It creates a 2x2 grid of subplots in a figure to visualize the training and testing performance over epochs, providing insights into the model's learning progress.
    
![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

## Summary
The Net model implemented in PyTorch provides an effective solution for classifying handwritten digits from the MNIST dataset. 
Its combination of convolutional and fully connected layers allows it to learn and recognize intricate patterns in the input images, enabling accurate digit classification.

## Next Steps
The Net model, designed for classifying handwritten digits from the MNIST dataset, consists of a total of 593,200 parameters. While this parameter count is relatively high, there are opportunities to enhance the model's accuracy and generalization. One approach is to experiment with different network architectures or regularization techniques, which can help optimize the model's performance by reducing overfitting and improving its ability to generalize to unseen data.
