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
...
