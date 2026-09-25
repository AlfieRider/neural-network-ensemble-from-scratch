# Activation Functions
Activation functions are simply mathematical functions, which are applied to the output of a neuron/neural-network layer, mapping this to a new output.

A rather key aspect about these is that they introduce non-linearity into the network, i.e. the output of a neuron will not change in direct proportion to its inputs, such that the models being produced from training are capable of learning more curved patterns, rather than being limited to linear patterns.

In other words, the lack of these would make stacking multiple layers end up ultimately behaving like that of a single linear transformation.

## Position in the Process:
More of a filler recall section (specifically from `neural-networks.md` and `forward propagation.md`).

The activation function is applied after the weighted sum calculation (during forward propagation). This is displayed below:

$$ \mathbf{z} = W\mathbf{x} = \mathbf{b} $$
$$ \mathbf{a} = f(\mathbf{z}) $$
> - `x` is the input to the layer
> - `W` is the weight matrix
> - `b` is the bias vector
> - `z` is the weighted sum prior to activation
> - `a` is the activated output

For an individual neuron, the process is shown below also:

$$ z = \mathbf{w}^T\mathbf{x} + b $$
$$ a = f(z) $$

## Actual Purpose of Activation Functions
...yea i like that subheading
