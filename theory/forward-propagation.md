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

## undecided title. decided content.
...
