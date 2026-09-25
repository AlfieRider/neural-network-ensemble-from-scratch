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
Linearity in terms of functions is simply a function with a linear relationship, obeying the properties associated with linear transformations. Such as:

$$ f(x) = 2x $$

which is also visually shown below:

```mermaid
xychart-beta
    title "Linear Function"
    x-axis "Input" [0, 1, 2, 3, 4, 5, 6]
    y-axis "Output" 0 --> 15
    line [0, 2, 4, 6, 8, 10, 12]
```

Any network consisting purely of linear transformations can be ultimately collapsed/condensed into one singular linear transformation.

An activation function, such as the following, is not linear:

$$ f(x) = max(0, x) $$

either mapping the input `x` to itself, or 0.
> this function `f` will only an input `x` to 0 if `x < 0` (i.e. the max of the two is 0).
> e.g. `f(-2) = 0`, whilst `f(2) = 2`.

Non-linearity such as this allows a network to construct increasingly complicated transformations as layers are composed/stitched/stuck together.

## Note on Applying Activation Functions in Practice
When working on an entire layer, the activation function itself is generally applied separately to each element of **z**. An example is given below:

$$
z =
\begin{bmatrix}
-2 \\
3 \\
-1 \\
5
\end{bmatrix}
$$

then, in this example, ReLU is applied (defined below):

$$ f(x) = max(0, z) $$

such that it follows:

$$
f(z) =
\begin{bmatrix}
f(-2) \\
f(3) \\
f(-1) \\
f(5)
\end{bmatrix}
$$

thus:

$$
\mathbf{a} =
\begin{bmatrix}
0 \\
3 \\
0 \\
5
\end{bmatrix}
$$

Note that the activation function has no impact on the shape of the vector;

If:

$$ \mathbf{z} \in ℝ^4 $$

then:

$$ \mathbf{a} \in ℝ^4 $$

The exact same also applies to when matrices process batches of example/input data (explored in `forward-propagation.md`.

## ReLU:
The Rectified Linear Unit (abbr. ReLU) is very commonly used. It is defined as simply as below:

$$ f(x) = max(0, x) $$

such that:

$$ f(x) =
\begin{cases}
0 & x < 0 \\
x & x \geq 0
\end{cases}
$$

ReLU example cases:

$$ f(-4) = 0 $$
$$ f(-0.2) = 0 $$
$$ f(0) = 0 $$
$$ f(2) = 2 $$
$$ f(7) = 7 $$

Simply put, ReLU removes negative values, whilst leaving positive values unchanged.

## The Derivative of ReLU
The derivative of ReLU (`f(x)`) is shown below:

$$
f`(x) =
\begin{cases}
0 & x < 0 \\
1 & x \geq 0
\end{cases}
$$
> note that the case `x = 0` is not accounted for, as the function itself has a corner.
> in implementation, a particular convention is chosen for this point.

For an active ReLU neuron, the local derivative is 1, thus the gradient can pass through without being multiplied by a small derivative. Whereas for a neuron whose input is negative, the derivative is 0, such that the gradient through ReLU is also 0.

This listed behaviour is a reason for ReLU being popular in gradient-based optimisations in comparison to other activation functions. More detail on this later, in `not sure yet...`.


