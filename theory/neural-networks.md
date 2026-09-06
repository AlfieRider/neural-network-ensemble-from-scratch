## What is a Neural Network?
- Machine Learning models composed of interconnected computational units. Loosely inspired by biological NN.
- Takes raw data, and learns patterns/relationships from it, without explicit pre-programmed rules dictating these patterns.
- Numerical parameters are adjusted to do this. These patterns, once learnt, can then be used to predict trends / make decisions on data.

In other words, a NN learns a function which maps an input onto an output.
```Input -> Neural Network -> Output```
The NN's ability to actually perform this mapping is contributed to many factors, including its architecture, parameters, activation functions, and the training process itself.

### Key Aspects:
#### Neurons
The fundamental computational units of a neural network.

A neuron receives 1/+ (one or more) inputs, combining these using learned `Weights` and a `Bias`, passing the result through an `activation function`.

Equations relating to neurons:

```z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b```

```a = f(z)```

Variables of ^^^ equations:
- `x` = input
- `w` = weight
- `b` = bias
- `z` = weighted sum (prior activation). `see linear transformation`
- `f` = activation function
- `a` = neuron output

### Learning:
A very simple procedure of:
1. Feed data into network
2. Network generates an output based on current parameters.
3. Network refines output by adjusting Ws and Bs such that performance is gradually improved.

## Layers in Neural Network Architecture:
- Input layer: network recieves the input data, where each input neuron corresponds to a specific feature in the given input data.
- Hidden layer/s: perform the most computational work; can have 1/n of these layers. Neurons tranform inputs to usable data for the output layer.
- Output layer: final layer providing the model's output. The format of such varies based on the task (e.g. classification, regression).

## Working of Neural Networks:
### 1. Forward Propagation:
This is when the data is input; data passes through the network in the forward direction (In->Hidden->Out).

1.1. Linear Transformation; each neuron in the layer recieves inputs, multiplied by associated connection's weights. These products are arithmetically summed, with a bias added on at the end.
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b 
```
^^^ w = weight, x = input, b = bias

1.2. Activation; result of linear transformation (z - see above) is passed through this activation function, which introduces non-linearity into the system, thus the network can learn more complex patterns.
> E.g: ReLU, sigmoid, tanh.

### 2. Backpropagation:
Comes after forward propagation; the network evaluates its performance using a loss function, which is used to measure the specifically the difference between the actual output and the predicted output. Training aims to minimise this loss.
- Loss Calculation: the network calculates the loss, providing a measure of error in the predictions. The loss function could also vary; common choices include mean squared error (for regression tasks) or cross-entropy loss (for classification).
- Gradient Calculation: Network computes gradients of the loss function with respect to network weight and network bias. Involves applying the chain rule to determine how much each part of the output error can be attributed to each weight and bias.
- Weight update: post-gradient calculation, the weights and biases are updated via an optimisation algorithm (e.g. SGD). The weights are adjusted in the opposite direction of the gradient to minimise the loss. The size of the step taken in each update is determined by the learning rate.

### 3. Iteration:
The above described process is repeated over many iterations over the entire dataset. This iteration in turn reduces the loss and the network's predictions become more accurate. Parameters can be adapted better to approximate relationships in the data, thus improving the overall performance for predictive modelling, etc.

## Sources:
will format properly later...
https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/
