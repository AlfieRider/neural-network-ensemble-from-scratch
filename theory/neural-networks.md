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

#### Connections:
- Link neurons between layers, allowing output of one neuron to become input of another neuron.
- Associated with weights, used to determine the strength and influence of an output.

#### Weights and Biases:
- The primary learnable parameters (i.e. change automatically during training for learning; improves performance).
- `Weights`: determine how strongly an input contributes to a neuron's output.
- `Biases`: provide an additional adjustable value, such that a neuron can shift its response independently of its inputs.

#### Activation Functions:
- Determines how a neuron's weighted input is transformed before being passed onward.
- Common examples: ReLU, Sigmoid, Tanh.
- Introduce non-linearity into the network; can learn relationships that cannot be linearly represented. (by a linear model)

#### Propagation:
- Information undertakes this process through the network as neurons perform their transformations.
- `Forward Propagation`: information moves from `Input -> Output`.
- `Backpropagation`: information about the error is propagated backward throughout the network; allows the network to determine how the parameters contributed to said error.

#### Learning Rule:
- Describes how parameters of the network are adjusted during training.
- `Gradient-based optimisation`: most modern NNs use this; gradients of the loss, w.r.t the network's parameters, are used to determine how the parameters should change.

### Learning:
A very simple procedure of:
1. Feed training data into network
2. Network generates an output based on current parameters.
3. Determine how different current output is to desired output.
4. Calculate how parameters contributed to this error
5. Network refines output by adjusting Ws and Bs such that performance is gradually improved.
6. Iterate.

`i.e. *Data -> Output -> Loss -> Gradients -> Param Update -> *` Iteratively.

## Layers in Neural Network Architecture:
Each layer, consisting of neurons, transforms the representation of the data it recieves.

'Input Layer -> Hidden Layer(s) -> Output Layer`

### Input Layer:
- Receives the original input data.
- Each input value represents a feature of the input.

Example:
- A model receiving information about a house, with values such as:
    - x₁: house size
    - x₂: bedroom total number
    - x₃: location
- For image classification (woo!) the input features may represent pixels.

### Hidden Layer(s):
- Sit between the input and output layers.
- Transform the representation of the input through successive (mathematical) operations. Can have 1/+ of these layers.
- `Deep Neural Network`: when a neural network contains multiple layers of learned transformations.

### Output Layer:
- Produces the networks final output.
- Format of such varies based on task itself.

Examples:
- `Binary Classification`: usually produces a single output representing one of two classes.
- `Multi-class Classification`: usually produces an output corresponding to each possible class available.
- `Regression`: produces 1/+ (one or more) continuous numerical values.

## Working of Neural Networks:
### 1. Forward Propagation:
This is when the data is input; data passes through the network in the forward direction (In->Hidden->Out).

#### 1.1. Linear Transformation: 
Each neuron in the layer recieves inputs, multiplied by associated connection's weights. These products are arithmetically summed, with a bias added on at the end.
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b 
```
^^^ `z` = result of linear/affine transformation, `w` = weight, `x` = input, `b` = bias

For an entire layer, this can be generalised using vectors and matrices, becoming `z = Wx + b`.
^^`W` represents the layer's weights, and `b` its biases.

#### 1.2. Activation 
The result of linear transformation (z - see above) is passed through an activation function (`a = f(z)`), which introduces non-linearity into the system, thus the network can learn more complex patterns.
> E.g: ReLU, sigmoid, tanh.

### 2. Backpropagation:
Comes after forward propagation; the network evaluates its performance using a loss function, which is used to measure the specifically the difference between the desired output (y) and the network-predicted output (ŷ).

`Compare(y, ŷ) -> Loss`

Training aims to minimise this loss.
> E.g.: Mean Squared Error (common for regression),
>       Cross-Entropy loss (common for classification)

#### 2.1 Gradient Calculation:
Backpropagation calculates the gradients of the loss w.r.t the network's parameters. These gradients, simply put, describe how much changing each parameter would in turn affect the loss produced.

To achieve this, apply the chain rule (of differentiation) through the network.

`Loss -> Output params -> Earlier params -> ... -> Input-side params`

The exact mathematics behind this principle are covered here: link will go here :)

2.2 Parameter Update:
The calculated gradients are used by an optimisation algorithm to update the network's parameters.

`parameterNew = parameterOld - learningRate*Gradient`

^^^where the `learning rate` determines the effective size of each update.

The parameters are in turn adjusted in the direction expected to reduce the loss. Further discussion of this can be found here: link to go here.

### 3. Iteration and Training:
The above described process of forward-pass, loss-calculation, backpropagation, and parameter-update, is repeated over many iterations over the entire dataset. This iteration in turn reduces the loss and the network's predictions become more accurate. Parameters can be adapted better to approximate relationships in the data, thus improving the overall performance for predictive modelling, etc.

Training itself does not guarantee that the network will perform well on unseen data; a network may instead learn training data too closely - this is called overfitting. See Generalisation and Overfitting link here for further exploration.

## Training vs Inference:
Training: the network's parameters are updated. This occurs after backpropagation, as discussed above.

Inference: The trained network itself uses its learned parameters to produce predictions for new inputs. The parameters are not usually updated during this.

## What is actually learnt?
Any Neural Network will likely not learn a collection of human-set and human-readable rules.

Training instead modifies its parameters, such that the network approximates a useful function relating its inputs to its output.

E.g. Image Classification: the network may learn increasingly useful representations of visual patterns as information is passed through the successive layers.

The exact meaning of individuals weights, or neurons themselves, is not necessarily straightforward to interpret however. Abstractly-put, a trained network can be represented as a function such as below:

`ŷ = f(x; θ)`

^^where `x` = input, `ŷ` = prediction, `θ` = network's learned parameters.

Training itself aims to find parameters that minimises the chosen loss function.

## Further Theory:
links here once written!!!

## Sources:
https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/

https://cs229.stanford.edu/summer2020/cs229-notes-deep_learning.pdf

https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html



