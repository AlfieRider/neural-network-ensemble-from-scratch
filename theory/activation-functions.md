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
This follows from the idea written above, that without activation functions, the transformations between layers would become a single linear transformation.

Consider two layers with no activation function between them, defined as below:

$$ \mathbf{h} = W_1\mathbf{X} + \mathbf{b}_1 $$
$$ \mathbf{y} = W_2\mathbf{X} + \mathbf{b}_2 $$

Through substitution of the first equation into the second, the following is obtained:

$$ \mathbf{y} = W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) $$
$$ \mathbf{y} = W_2W_2\mathbf{x} + W_2\mathbf{b}_1 + \mathbf{b}_2 $$

which can be written in the form:

$$ \mathbf{y} = W′\mathbf{x} + \mathbf{b}′ $$

for some new W′ and **b**′.

And this is not just limited to two layers; regardless of how many layers are chained together without activation functions, the result is always the same, producing a result that could've been produced through one single matrix multiplication.

Through implementation of activation functions, this property is in turn broken, such that non-linearity is introduced between layers where it otherwise would not have existed.

## Further Mathematical Point on Non-Linearity's definition:
as it could be useful to have some more explanation
