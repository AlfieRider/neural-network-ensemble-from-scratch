# Forward Propagation
Forward Propagation, as written in `neural-networks.md` is the process through which a neural network takes input, passes this input through its layers, then produces an output. The following sequence is iteratively performed by any forward pass:

1. Receive the input values entering the given layer.
2. Multiply these by the layer's weights.
3. Add the layer's biases to ^^.
4. Apply an activation function.
5. Pass the resulting activations to the next layer.

In other words, a formula for ^^ can be expressed as shown (per layer):

$$
\mathbf{a} = f(W\mathbf{x} + \mathbf{b})
$$
> - **x** is the input to the specified layer
> - W is the layer's weight matrix
> - **b** is the layer's bias vector.
> - W**x** + **b** produces the pre-activation values.
> - f is the activation function (e.g. ReLU, tanh, etc)
> - **a** is the layer's output, i.e. `activiation`.

Readers may recognise the above (^^) from `mathematical-foundations.md`. This process, as mentioned, is repeatedly applied throughout the network as data moves through it.

## The Neuron Model:
Consider a neuron receiving the following inputs:

$$
x_1, x_2, x_3
$$

then the same neuron's corresponding weights:

$$
w_1, w_2, w_3
$$

The pre-activation calculation is as follows:

$$
z = w_1x_1 + w_2x_2+ w_3x_3 + b
$$

The neuron then applies an activation function, such that:

$$
a = f(z)
$$

Using vector notation (introduced in `mathematical-foundations.md`), this is expressed as:

$$
a = f(\mathbf{w}^T\mathbf{x} + \mathbf{b})
$$

## Forward Propagation through a Layer:
Each layer is expected to contain a multitude of neurons.

Suppose the given input contains three values, such that:

$$
x \in \mathbb{ℝ}^3
$$

and suppose that the layer in question contains 4 neurons.

Each neuron needs its own set of three weights; each neuron receives all 3 input values (weights * input). Therefore, the weight matrix in this case will have four rows and three columns, as shown:

$$
x \in \mathbb{ℝ}^{4 \times 3}
$$

> thus note that the general formula for the shape of the Weight matrix is:
> W => (numNeurons, numInputs)

This entails that

$$
W \times \mathbf{x}
$$

produces four values, one per neuron.

The bias vector also needs one value per neuron, such that:

$$
b \in \mathbb{ℝ}^4
$$

In turn, the complete pre-activation calculation is consequently:

$$
\mathbf{z} = W\mathbf{x} + \mathbf{b}
$$

with:

$$
\mathbf{z} \in \mathbb{ℝ}^4
$$

The activation function is then applied to each element, as shown:

$$
\mathbf{a} = f(\mathbf{z})
$$

which then gives:

$$
\mathbf{a} \in \mathbb{ℝ}^4
$$

The dimensionality can be seen to change from three input values to four output values, following the below:

$$
\boxed{\mathbf{x_4} \rightarrow W_{4\times3} \rightarrow \mathbf{z_4} \rightarrow \mathbf{a_4}}
$$
> Note that the subscript here is used to indicate the number of elements, rather than the formal notation for the vectors themselves.

## Matrix Multiplication Insight
This section will aim to clarify (/recap, for readers of `mathematical-foundations.md`), the frequently occurring operation, which is displayed below:

$$
W\mathbf{x}
$$
> The key takeaway from the following section is that the weighted sum for every neuron is performed simultaneously.
> Do feel free to skip to the next section!

Example; suppose:

$$
W =
\begin{bmatrix}
\ w_{11} & w_{12} & w_{13} \\
\ w_{21} & w_{22} & w_{23} \\
\ w_{31} & w_{32} & w_{33} \\
\ w_{41} & w_{42} & w_{43}
\end{bmatrix}
$$

and that:

$$
\mathbf{x} =
\begin{bmatrix}
\ x_1 \\
\ x_2 \\
\ x_3
\end{bmatrix}
$$

The following calculation occurs:

$$
W\mathbf{x}
$$

$$
= \begin{bmatrix}
\ w_{11}x_1 + w_{12}x_2 + w_{13}x_3 \\
\ w_{21}x_1 + w_{22}x_2 + w_{23}x_3 \\
\ w_{31}x_1 + w_{32}x_2 + w_{33}x_3 \\
\ w_{41}x_1 + w_{42}x_2 + w_{43}x_3
\end{bmatrix}
$$
> recall that the shape of Wx is derived from (4x3)(3x1) => (4,1)

Each row produced from ^^ represents one neuron's weighted sum, thus proves that the matrix itself allows the network to calculate for all four given neurons simultaneously.

After then adding the bias vector:

$$
\mathbf{z} =
\begin{bmatrix}
\ w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1 \\
\ w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2 \\
\ w_{31}x_1 + w_{32}x_2 + w_{33}x_3 + b_3 \\
\ w_{41}x_1 + w_{42}x_2 + w_{43}x_3 + b_4
\end{bmatrix}
$$

each element then therefore becomes the pre-activation of a specific given neuron.

## Applying the Activation Function
Yet again serves as another clarification exercise, or alternatively a `mathematical-foundations.md` recap.

The activation function is applied after the weighted sum and bias have been calculated.

For example, assuming the layer uses ReLU, defined as follows:

$$
\mathbf{f}(z) = max(0, z)
$$

and the given pre-activation vector is such that:

$$
\mathbf{z} =
\begin{bmatrix}
2 \\
-1 \\
4 \\
-3 \\
\end{bmatrix}
$$

then:

$$
\mathbf{a} = \mathrm{ReLU}(\mathbf{z}) =
\begin{bmatrix}
2 \\
0 \\
4 \\
0 \\
\end{bmatrix}
$$
> note: the activation function is applied sequentially through each element.

The above thus explains the following shorthand, which represents applying f to every element of the vector `Wx + b`:

$$
\mathbf{a} = \mathbf{f}(W\mathbf{x} + \mathbf{b})
$$

## Layer Calculation in full:
Consider a layer taking three inputs and containing two neurons.

For the given inputs, let:

$$
\mathbf{x} =
\begin{bmatrix}
2 \\
3 \\
1
\end{bmatrix}
$$

and for weights, let:

$$
W =
\begin{bmatrix}
1 & 2 & -1 \\
-2 & 1 & 3
\end{bmatrix}
$$
> the shape of W is `2 x 3`, as it contains 2 rows (neurons) and 3 columns (inputs per neuron)

For the bias vector, let:

$$
\mathbf{b} =
\begin{bmatrix}
1 \\
-2
\end{bmatrix}
$$

Recall that the pre-activation is defined as below:

$$
\mathbf{z} = W\mathbf{x} + \mathbf{b}
$$

First, `Wx` is calculated:

$$
W\mathbf{x}
$$
$$
= \begin{bmatrix}
1 & 2 & -1 \\
-2 & 1 & 3
\end{bmatrix}
\begin{bmatrix}
2 \\
3 \\
1
\end{bmatrix}
$$
$$
= \begin{bmatrix}
1(2) + 2(3) - 1(1) \\
-2(2) + 1(3) + 3(1)
\end{bmatrix}
$$
$$
= \begin{bmatrix}
7 \\
2
\end{bmatrix}
$$

The bias vector is then added:

$$
\mathbf{z} =
\begin{bmatrix}
7 \\
2
\end{bmatrix} +
\begin{bmatrix}
1 \\
-2
\end{bmatrix}
$$
$$
= \begin{bmatrix}
8 \\
0
\end{bmatrix}
$$

Application of an activation function (ReLU used here) is shown below:

$$
\mathbf{a} = \mathop{\text{ReLU}}
\begin{bmatrix}
8 \\
0
\end{bmatrix} =
\begin{bmatrix}
8 \\
0
\end{bmatrix}
$$
> recall that ReLU = max(0, z)

Therefore the output of this layer is:

$$
\mathbf{a} = 
\begin{bmatrix}
8 \\
0
\end{bmatrix}
$$

## Forward Propagation through Multiple Layers:
The real fun starts when these individual layers are strung together. This is best explained through a simplified example.

Suppose a network has:
- 3 input values
- 4 neurons in Layer 1
- 2 neurons in Layer 2
- 1 output neuron.

The first layer recieves:

$$
\mathbf{x} \in ℝ^3
$$
> recall this notation from earlier in this document.

and through calculation, the first layer then produces:

$$
\mathbf{x}^{(1)} \in ℝ^4
$$
> i.e. vector **x** in the 1st Layer produces 4 real values, such that {x₁, x₂, x₃, x₄} belongs to ℝ^4.

The second layer therefore receives these  4 input values.

The weight matrix consequently has 4 columns, as dictated below:

$$
W^{(2)} \in ℝ^{2\times4}
$$
> i.e. W contains two neurons, each receiving 4 inputs.

The second layer then produces:

$$
\mathbf{x}^{(2)} \in ℝ^2
$$

The final layer then in turn receives both of these `a` values, thus the weight matrix has shape:

$$
W^{(3)} \in ℝ^{1\times2}
$$

and produces one value.

The overall structure here is:

$$
3 \rightarrow 4 \rightarrow 2 \rightarrow 1
$$

or in vector terms:

$$
\mathbf{x} \rightarrow \mathbf{a}^{(1)} \rightarrow \mathbf{a}^{(2)} \rightarrow \mathbf{a}^{(3)}
$$

## Equations / Calculations for Multiple Layers:
For the first layer:

$$
\mathbf{z}^{(1)} = W^{(1)}\mathbf(x) + \mathbf{b}^{(1)}
$$
$$
\mathbf{a}^{(1)} = f^{(1)}(\mathbf{z}^{(1)})
$$

The first layer's output becomes the second layer's input. Therefore:

$$
\mathbf{z}^{(2)} = W^{(2)}\mathbf{a}^{(1)} + \mathbf{b}^{(2)}
$$
$$
\mathbf{a}^{(2)} = f^{(2)}(\mathbf{z}^{(2)})
$$

A generalised formula for this is such that:

$$
\mathbf{a}^{(n+1)} = f^{(n+1)}(W^{(n+1)}\mathbf{a}^{(n)} + \mathbf{b}^{(n+1)})
$$

such that the final activation produced is the network's output.

## Trans-Layer Dimension Change:
The dimensions of each weight matrix are determined by the number of inputs and neurons in that particular layer.

Suppose that the first layer contains 4 neurons, and that:

$$
\mathbf{x} \in ℝ^{3}
$$
> such that the vector has 3 rows and 1 column (i.e. shape of 3x1)

Such that also:

$$
W^{(1)} \in ℝ^{4\times3}
$$
> four neurons, 3 inputs.

Given the above, it follows that:

$$
(4x3)(3x1) = (4x1)
$$

Therefore it can be concluded that `W¹**x**` produces 4 values (presented as 4 rows, 1 column).

Now consider that the four values produced ^^ are the input to the next layer:

$$
\mathbf{a}^{(1)} \in ℝ^4
$$
> i.e. a⁽¹⁾ here is the input of four values produced from layer 1.

If the next layer contains, say, 2 neurons, such that:

$$
W^{(2)} \in ℝ^{2\times4}
$$
> i.e. ℝ<sup>(2 neurons x 4 inputs)</sup>

as:

$$
(2\times4)(4\times1) = (2\times1)
$$

and in turn the result contains 2 values (presented in one column), such that:

$$
\mathbf{a}^{(2)} \in ℝ^2
$$

The fundamental relationship is written below:
> The number of columns in a layer's Weight matrix `W` must be equivalent to the number of values entering that layer, whilst the number of rows in `W` is equivalent to the number of neurons in said layer.

## Forward Propagation as Function Composition:
A neural network can also be understood as a sequence of functions. A layer performs:

$$
f(W\mathbf{x} + \mathbf{b})
$$

So a two layer network may in turn be represented as:

$$
a^{(1)} = f_1(W_1\mathbf{x} + \mathbf{b_1}
$$

followed by:

$$
a^{(2)} = f_2(W_2\mathbf{x} + \mathbf{b_2}
$$

Through substitution of the Layer1 equation into the Layer2 equation, the following representation is produced:

$$
a^{(2)} = f_2(W_2f_1(W_1\mathbf{x} + \mathbf{b_1}) + \mathbf{b_2})
$$

which conceptually is simply:

$$
\mathbf{x} \rightarrow \text{Layer 1} \rightarrow \text{Layer 2} \rightarrow \text{Output}
$$

Each layer transforms its input into a new representation, which is then transformed by the following layer. This is why neural networks can be mathematically viewed as compositions of functions.

# A Multi-Layer Process:
Below is an example of how a network will function via multiple layers. This is conceptually the same as earlier in this document (and as in `mathematical-foundations.md`), however a real-value example may be worth including.

Consider a network with:
- 2 input values
- 2 neurons in the hidden layer
- 1 output neuron

This will be diagrammatically represented as such:

$$
2 \rightarrow 2 \rightarrow 1
$$

Let:

$$
\mathbf{x} =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
$$
$$
W^{(1)} =
\begin{bmatrix}
2 & 1 \\
-1 & 3
\end{bmatrix}
$$
$$
\mathbf{b}^{(1)} =
\begin{bmatrix}
1 \\
-2
\end{bmatrix}
$$

The first layer calculates the following:

$$
\mathbf{z}^{(1)} = W^{(1)}\mathbf(x) + \mathbf{b}^{(1)}
$$
$$
\mathbf{z}^{(1)} =
\begin{bmatrix}
2(1) + 1(2) + 1 \\
-1(1) + 3(2) - 2
\end{bmatrix}
$$
$$
= \begin{bmatrix}
5 \\
3
\end{bmatrix}
$$

An through applying ReLU for activation gives:

$$
\mathbf{a}^{(1)} =
\begin{bmatrix}
5 \\
3
\end{bmatrix}
$$

This final vector for the 1st Layer is then passed as input into the 2nd Layer. Suppose that:

$$
W^{(2)} =
\begin{bmatrix}
2 & -1
\end{bmatrix}
$$
$$
\mathbf{b}^{(2)} = 1
$$

Then the following calculation occurs:

$$
\mathbf{z}^{(2)} = W^{(2)}\mathbf{a}^{(1)} + \mathbf{b}^{(2)}
$$
$$
= 2(5) - 1(3) + 1
$$
$$
= 8
$$
> note that we are left with a 1x1 shape answer; (2x1)(1x2) = (1x1)

In turn, the network has transformed the input vector into a single output value through a sequence of learned transformations, as simplified below:

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix}
\rightarrow
8
$$

## The Output Layer
The final layer of the neural network is structurally identical to any other layer, such that the below still applies:

$$
\mathbf{z}^{(L)} = W^{(L)}\mathbf{a}^{(L-1)} + \mathbf{b}^{(L)}
$$

However, it is so crucial to note that the activation function used by this layer depends solely on the task at hand.

For example, a regression network may produce a continuous value directly:

$$
ŷ = z
$$

A binary classification network may use a sigmoid:

$$
ŷ = σ(z)
$$
> ideal for probabilities, as it maps the input `z` to a number `n` such that {n: 0 ≤ n ≤ 1}

A multi-class classification may use softmax:

$$
ŷ = \text{softmax}(z)
$$
> all inputs are converted into a probability distribution, such that all values sum to 1.

Where the output `ŷ` is representative of the network's prediction (i.e. output).

The choice of output activation is therefore strongly connected to the type of problem being solved.

## Forward Propagation in an Image Classifier:
Consider now the purpose of this project (repo): a neural network trained to classify handwritten digits.

The input may, for example, contain 784 values, corresponding to a 28 x 28 image flattened into a vector. See the input defined below:

$$
\mathbf{x} \in ℝ^{784}
$$

The first hidden layer may contain 128 neurons:

$$
W^{(1)} \in ℝ^{128 \times 784}
$$

in turn producing:

$$
\mathbf{a}^{(1)} \in ℝ^{128}
$$

Following this, the second hidden layer (in this example) may contain 64 neurons, such that:

$$
W^{(2)} \in ℝ^{64\times128}
$$

thus producing:

$$
\mathbf{a}^{(2)} \in ℝ^{64}
$$

Finally, the output layer contains 10 neurons, one corresponding to each digit from 0 to 9, as shown:

$$
W^{(3)} \in ℝ^{10\times64}
$$

which produces:

$$
\mathbf{z}^{(3)} \in ℝ^{10}
$$
> note that this is pre-activation.

Softmax can then convert these 10 values into probabilities:

$$
ŷ = \text{softmax}(\mathbf{z}^{(3)})
$$

An example output to explain this is shown below:

$$
ŷ =
\begin{bmatrix}
0.01 \\
0.02 \\
0.03 \\
0.01 \\
0.05 \\
0.02 \\
0.81 \\
0.01 \\
0.03 \\
0.01
\end{bmatrix}
$$

As the largest probability is associated with index 6, the network predicts that 6 is the output. In other summative words, this forward propagation has transformed the raw image itself into this prediction.

## Python Implementation of Forward Propagation:
This is not to be interpreted as design of the solution itself, rather an example of how this can be programmed for further conceptual understanding. Potentially can be considered as theory of the design.

The expression:

$$ \mathbf{z} = W\mathbf{x} + \mathbf{b} $$

maps effectively directly into code via NumPy, shown below (assume import):

```
z = W @ x + b
```
> note: "@" is the operator for matrix multiplication here.

The activation function can be just as easily applied (following the above):

```
a = relu(z)
```

A full python example is written below, for completion's sake:

```
import numpy

def relu(x):
    return np.maximum(0, x)

def forwardPass(x, W, b):
    z = W @ x + b
    a = relu(z)
    return a
```

## Forward Propagation in Training:
Forward propagation naturally occurs through both training and inference. As covered in `neural-networks.md`, the training process is roughly as shown:

$$ Input \rightarrow Forward Propagation \rightarrow Prediction \rightarrow Loss \rightarrow Backpropagation \rightarrow Parameter update $$

The forward pass then uses the network's current weights and biases to produce a prediction, `ŷ`.

As this is in **training**, the prediction `ŷ` is then compared with the expected/desired output via a loss function, such as dictated simply below:

$$ L(ŷ, y) $$
> where:
> - `ŷ` is the network's output
> - `y` is the expected/desired output
> - `L` is the loss function itself

It is then **back propagation** which determines how changing the network's parameters would affect this loss. Optimisation then uses the resulting gradients to update the parameters, adjusting the network closer to producing the desired output for input **x**.

## Forward Propagation in Inference:
Inference, as a reminder, is simply the network producing an output outside of training processes; once a network has been trained, forward propagation can be used on previously unseen data. In other words, the weights and biases are no longer updated.

Forward propagation therefore does not learn anything new here, rather instead is used purely as the mechanism for using the current network state to produce an output.

## Batches of Inputs:
To introduce something new:

The examples and equations above so far have been limited to considering one input at a time. In practice however, neural networks generally process multiple examples together in a batch.

Suppose there exist 32 examples of input batches, each containing 3 input values. Above, this would be represented as simply:

$$ \mathbf{X} \in ℝ^3 $$

Instead however, the batch can be represented as a matrix, such as defined below:

$$ X \in ℝ^{32 \times 3} $$

where each row in turn represents one batch.

Given a layer has, for demonstration's sake, 4 neurons, the weight matrix is still:

$$ W \in ℝ^{4 \times 3} $$
> one row per neuron, and 3 (input) values per (neuron) row

Using the row-focussed representation:

$$ XW^T $$

which has dimensions of:

$$ (32 \times 3)(3 \times 4) = 32 \times 4 $$

i.e. the result contains four outputs for each of the 32 batches.
> which corresponds to each of the 4 neurons' outputs in each single batch.
> such that the result has 32 rows and 4 columns for this.

The bias vector, defined as:

$$ \mathbf{b} \in ℝ^4 $$

is added then after to each row.

A summary of this process is in the following equations:

$$ Z = XW^T + \mathbf{b} $$
$$ A = f(Z) $$
> f is the activation function

Through this approach, batches can be processed much more efficiently.

## Summary:
Forward propagation is the process of passing an input through a neural network to produce a prediction.

For an individual layer, the fundamental operation that occurs is defined as:

$$ a = f(Wx + b) $$
> 1.1 Weighted Sum: `z = Wx`
> 1.2 Added biases: `z = Wx + b`
> 2. Applied activation function: `a = f(z)`

Multiple layers repeat this process, with the output of one layer becoming the input of the next. See the below:

$$ a^{(1)} = f^{(1)}(W^{(1)}x + b^{(1)}) $$
$$ a^{(2)} = f^{(1)}(W^{(2)}a^{(1)} + b^{(2)}) $$
> etc, continuing to the output layer.

## Sources:
- https://cs231n.github.io/neural-networks-1/
- https://www.geeksforgeeks.org/deep-learning/forward-propagation-in-neural-networks/
- https://www.datacamp.com/tutorial/forward-propagation-neural-networks
- https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-33-neural-nets-and-the-learning-function
- https://www.deeplearningbook.org/contents/mlp.html
