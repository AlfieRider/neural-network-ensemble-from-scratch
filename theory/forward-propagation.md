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

## Equations \ Calculations for Multiple Layers:
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

ugh
