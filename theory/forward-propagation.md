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
x \in \mathbb{R}^3
$$

and suppose that the layer in question contains 4 neurons.

Each neuron needs its own set of three weights; each neuron receives all 3 input values (weights * input). Therefore, the weight matrix in this case will have four rows and three columns, as shown:

$$
x \in \mathbb{R}^{4 \times 3}
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
b \in \mathbb{R}^4
$$

In turn, the complete pre-activation calculation is consequently:

$$
\mathbf{z} = W\mathbf{x} + \mathbf{b}
$$

with:

$$
\mathbf{z} \in \mathbb{R}^4
$$

The activation function is then applied to each element, as shown:

$$
\mathbf{a} = f(\mathbf{z})
$$

which then gives:

$$
\mathbf{a} \in \mathbb{R}^4
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
> recall that the shape of W**x** is derived from (4x3)(3x1) => (4,1)

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

