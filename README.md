
# Handwritten Digit Recognition using Convolutional Neural Networks with Pytorch

The MNIST dataset comprises 70,000 grayscale images of handwritten digits, each measuring 28 x 28 pixels and annotated by humans. It is a subset of a larger dataset made accessible by NIST, the National Institute of Standards and Technology. In this example, we will develop a handwritten digit recognizer utilizing a convolutional neural network (CNN) trained on the MNIST dataset, employing the PyTorch framework.

<p align="center">    
    <img width="500" aling="right" src="https://github.com/Paurnima-Chavan/MNIST-pytorch/blob/main/imgs/handwriiten.png?raw=true" />
</p>


## Documentation

This documentation provides an overview of the PyTorch model used for classifying handwritten digits from the MNIST dataset. The model architecture consists of convolutional neural networks (CNNs) and fully connected layers, which are designed to achieve accurate digit recognition.

## Code organization

Code organization in this project is structured into three files.

`models.py` 

`utils.py`

`S5.ipynb`

**"models.py"** contains the model class, **"utils.py"** includes code for training, testing, and generating performance graphs, while **"S5.ipynb"** serves as the notebook for the actual execution and experimentation. 

This separation allows for modular and organized development of the project components.

## Model Architecture

The model, implemented in the Net class, follows a sequential structure that defines the layers and operations performed on the input data.

```bash
    class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4
        x = x.view(-1, 4096) # 4*4*256 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

### Model Summary

| Layer (type) | Output Shape | Param #  |
| ------------- |-------------:| -----:|
| Conv2d-1 | [-1, 32, 26, 26] | 320 |
| Conv2d-2 | [-1, 64, 24, 24] | 18, 496 |
| Conv2d-3 | [-1, 128, 10, 10] | 73, 856 |
| Conv2d-4 | [-1, 256, 8, 8] | 295, 168|
| Linear-5 | [-1, 50] | 204, 850 |
| Linear-6 | [-1, 10]  | 510 |

Total params: 593,200

Trainable params: 593,200

Non-trainable params: 0
<hr class="dashed">

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
    ```bash
        # Train data transformations
        train_transforms = transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15., 15.), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])

        # Test data transformations
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
    ```
- Pass the preprocessed input through the model's forward method to obtain the predicted digit class probabilities.
- We have used SGD (Stochastic Gradient Descent) optimizer, which will be used to update the model's parameters during training. The learning rate is set to 0.01, and the momentum is set to 0.9
- Trained the model for the specified number of epochs, printing the epoch number, and performs training and testing steps in each epoch. The learning rate is adjusted using the scheduler, allowing better optimization over time. The train and test functions are responsible for the actual training and testing processes, respectively.
    ```bash
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
        # New Line
        criterion = F.nll_loss 
        num_epochs = 20

        for epoch in range(1, num_epochs+1):
          print(f'Epoch {epoch}')
          train(model, device, train_loader, optimizer, criterion)
          test(model, device, test_loader, criterion)
          scheduler.step()
    ```
- Finally, plotted the training and testing accuracy as well as the training and testing loss. It creates a 2x2 grid of subplots in a figure to visualize the training and testing performance over epochs, providing insights into the model's learning progress.
<p align="center">    
    <img width="800" hight="300" aling="right" src="https://github.com/Paurnima-Chavan/MNIST-pytorch/blob/main/imgs/performance.png" />
 </p>

## Summary
The Net model implemented in PyTorch provides an effective solution for classifying handwritten digits from the MNIST dataset. 
Its combination of convolutional and fully connected layers allows it to learn and recognize intricate patterns in the input images, enabling accurate digit classification.

## Next Steps
The Net model, designed for classifying handwritten digits from the MNIST dataset, consists of a total of 593,200 parameters. While this parameter count is relatively high, there are opportunities to enhance the model's accuracy and generalization. One approach is to experiment with different network architectures or regularization techniques, which can help optimize the model's performance by reducing overfitting and improving its ability to generalize to unseen data.
