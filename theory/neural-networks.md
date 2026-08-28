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

## Workign of Neural Networks:
1. Forward Propagation:
    This is when the data is input; data passes through the network in the forward direction (In->Hidden->Out).
> 1.1. Linear Transformation; each neuron in the layer recieves inputs, multiplied by associated connection's weights. These products are arithmetically summed, with a bias added on at the end.
> `z = w1x1 + w2x2 + ... + wnxn + b`; w = weight, x = input, b = bias.
> 1.2. Activation; result of linear transformation (z - see above) is passed through this activation function, which introduces non-linearity into the system, thus the network can learn more complex patterns.
> E.g: ReLU, sigmoid, tanh.

3. Backpropagation:
