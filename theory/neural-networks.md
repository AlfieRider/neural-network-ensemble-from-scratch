## What is a Neural Network?
- ML models made to mimic human brain function.
- Models usually consist of connected nodes which process data in such a way that pattern recognition is plausible, such that the patterns learnt aren't necessarily influenced by pre-set rules.

### Key Aspects:
- Neurons: Node-equivalent. Recieve inputs. Each neuron is governed by a threshold and an activation function.
- Connections: Edge-equivalent. Links 2/+ nodes, and can carry information. Have weights and biases.
- Weight and Biases: oversee strength and influence of connections.
- Propagation Functions: Aid in processing and transferring data across neuron layers.
- Learning Rule: a method, such that weights and biases are adjusted over time to fit the desired result (improves accuracy).

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
z = w1x1 + w2x2 + ... + wnxn + b 
```
^^^ w = weight, x = input, b = bias

1.2. Activation; result of linear transformation (z - see above) is passed through this activation function, which introduces non-linearity into the system, thus the network can learn more complex patterns.
> E.g: ReLU, sigmoid, tanh.

### 2. Backpropagation:
Comes after forward propagation; the network evaluates its performance using a loss function, which is used to measure the specifically the difference between the actual output and the predicted output. Training aims to minimise this loss.
- Loss Calculation: the network calculates the loss, providing a measure of error in the predictions. The loss function could also vary; common choices include mean squared error (for regression tasks) or cross-entropy loss (for classification).
- Gradient Calculation: Network computes gradients of the loss function with respect to network weight and network bias. Involves applying the chain rule to determine how much each part of the output error can be attributed to each weight and bias.
- Weight update: post-gradient calculation, the weights and biases are updated via an optimisation algorithm (e.g. SGD). The weights are adjusted in the opposite direction of the gradient to minimise the loss. The size of the step taken in each update is determined by the learning rate.

## Sources:
will format properly later...
https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/
