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

Note that this is very similar to `Binary Step`. In fact, the derivative of ReLU above is written conventionally, where the case: 

$$
f`(x) =
\begin{cases}
0 & x < 0 \\
1 & x \geq 0
\end{cases}
$$

should mathematically instead be:

$$
f`(x) =
\begin{cases}
0 & x < 0 \\
1 & x > 0
\end{cases}
$$

thus an alternative conventional ReLU derivative definition can also instead state that:

$$
f`(x) =
\begin{cases}
0 & x \leq 0 \\
1 & x > 0
\end{cases}
$$

Mathematically, the truest derivative of ReLU is undefined at `x = 0`. The convention itself only matters if a neuron's pre-activation happens to be exactly 0.

For a ReLU neuron with a positive input, the local derivative is 1, thus the gradient can pass through without being multiplied by a small derivative. Whereas for a neuron whose input is negative, the derivative is 0, such that the gradient through ReLU is also 0.

This listed behaviour is a reason for ReLU being popular in gradient-based optimisations in comparison to other activation functions. More detail on this later, in `not sure yet...`.

## "Dying ReLU" Problem:
ReLU has an overarching weakness; a kind of heavenly restriction at a push.

Let z represented the summed weighted bias, and define z such that:

$$ z < 0 $$
> i.e. the case in which z from a neuron's output is always negative.

Consequently, the gradient passing through said activation can become zero.

If the neuron's parameters are subsequently updated in such a way that keeps the output negative for all training examples, the neuron itself effectively becomes redundant, in that it contributes nothing to the network.

This should not however be confused: every negative value will not permanently kill a neuron. Rather, any ReLU neuron can freely switch between active and inactive states as its input changes. The problem itself arises when a neuron becomes stuck in the inactive region during training itself.

## Leaky ReLU
Leaky ReLU is a modified version of ReLU, aimed to address some of the already described problems with ordinary ReLU.

Instead of completely removing negative values, Leaky ReLU allows a small negative output instead. It is defined below:

$$
f(X) =
\begin{cases}
αx & x < 0 \\
x & x \geq 0
\end{cases}
$$
> where α is a small positive constant.

For example, if `α = 0.01`, then:

$$ f(-5) = -0.05 $$

rather than `0`.

The negative side therefore has a small gradient, rather than a gradient of exactly 0.

## Sigmoid
This is also referred to as the logistic sigmoid function, defined as:

$$ σ(x) = \frac{1}{1 + e^{-x}} $$

mapping inputs into the following range:

$$ 0 < σ(x) < 1 $$

Some example values include:

$$ σ(-5) \approx 0.0067 $$
$$ σ(0) = 0.5 $$
$$ σ(5) \approx 0.9933 $$

Conceptually, to help with interpretation of the range:

$$ -\infty \rightarrow 0 $$
$$ 0 \rightarrow 0.5 $$
$$ +\infty \rightarrow 1 $$

![alt text](https://drek4537l1klr.cloudfront.net/chaudhury/Figures/CH01_F05_Chaudhury.jpg)

## Problems with Sigmoid
The key underlying issue with sigmoid is saturation. To explain this, consider very large positive or negative inputs.

For a large positive x:

$$ σ(x) \approx 1 $$

For a large negative x:

$$ σ(x) \approx 0 $$

Thus the function becomes very flat in both regions implied/described above.

On the topic of its derivative, shown below:

$$ σ`(x) = σ(x)(1-σ(x)) $$

Since the sigmoid output lies between 0 and 1, the derivative becomes very small when the output approaches either of these extremes.

For example, if:

$$ σ(x) \approx 1 $$

then it naturally follows that:

$$ σ`(x) \approx 1(1-1) $$
$$ = 0 $$

Similarly, given:

$$ σ(x) \approx 0 $$

then it naturally follows that:

$$ σ`(x) \approx 0(1-0) $$
$$ = 0 $$

During backpropagation, gradients are repeatedly multiplied by derivatives. In turn, if an activation contributes a very small derivative, then the gradient passing through it can become very small. This is another of many forms of the "vanishing-gradient problem".

## Tanh
The hyperbolic tangent, typically written as `tanh(x)` is another commonly known activiation function.

It maps real-valued inputs to (-1, 1). It is defined as below:

$$ tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

- For negative infinities, tanh will tend to `-1`.
- For 0, tanh is simply `0`.
- For positive infinities, tanh will tend to `1`.
> as shown here, tanh is zero-centered.

With interval (-1, 1), it can have equally positive and negative numbers.

![alt text](https://media.geeksforgeeks.org/wp-content/uploads/20250214171817652462/tanh.png)

## Fun Similarities - Sigmoid and Tanh:
Sigmoid and Tanh are mathematically related, shown by the below transformation:

$$ tanh(x) = 2σ(2x) - 1 $$

Similarities:
- are both smooth
- are both bounded
- both saturate for sufficiently large + and - inputs
- both can suffer from small gradients in their saturated regions.

Their key difference is that tanh is zero centered, whereas `σ(0) = 0.5`. Additionally, due to this, tanh more closely resembles the identity function around zero (i.e. that id(x) = x).

## Activation Functions & the Output Layer:
The activation function used in a network does not necessarily have to be the same for each layer. In particular, the output layer often will have a different requirement from hidden layers.

Consider a network which classifies an image into one of ten digits (self-awareness), from `0 -> 9`.

The final layer may produce these ten raw values:

$$
\mathbf{z} =
\begin{bmatrix}
2.1 \\
-0.7 \\
4.8 \\
1.2 \\
-2.3 \\
0.1 \\
0.8 \\
-1.4 \\
0.3 \\
0.5
\end{bmatrix}
$$

These values are often called logits, or alternatively class scores. These themselves do not represent probabilities.

For a multiclass classification problem, softmax can be applied, in turning these scores into a probability distribution.

## Softmax
Softmax is defined for a vector of values. For the i-th element:

$$
softmax(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j}^{J}e^{z_j}}
$$
> for i = 1, 2, ..., J

The result of this has two key properites:

$$ 0 < softmax(z)_i < 1 $$

and:

$$ \sum_{i} softmax(z)_i = 1 $$

Therefore the output can be interpreted as a probability distribution over classes.

For example:

$$
z =
\begin{bmatrix}
1 \\
2 \\
4
\end{bmatrix}
$$

may produce approximately:

$$
softmax(z) =
\begin{bmatrix}
0.0042 \\
0.114 \\
0.844
\end{bmatrix}
$$

where the largest logit corresponds to the largest probability. So a classifier such as this could thus interpret this as likelihoods of classes, such as:

- Class 1: 4.2%
- Class 2: 11.4%
- Class 3: 84.4%
> which sums also to 1.

# Special Property of Softmax:
Softmax is different from ordinary hidden-layer activations. ReLU, sigmoid, tanh are element-wise activations, where each value can be processed independently. For example:

$$
ReLU
\begin{pmatrix}
-2 \\
3 \\
5
\end{pmatrix} =
\begin{pmatrix}
0 \\
3 \\
5
\end{pmatrix}
$$

The output of softmax for each element however is dependant on all of the elemtns, as every demoninator contains:

$$ \sum_{j} e^{z_j} $$

which in turn means that through even changing one singular value, the entire output probability can change, which makes softmax particularily useful/appropriate when the outputs represent competing classes in a multiclass classification problem.

# Binary Classifcation & Sigmoid:
For binary classification, only one output value is required, representing the probability of one of two classes.

For example, say:

$$ a = \sigma (z) $$

may produce the following output:

$$ a = 0.87 $$

which can be in turn interpreted as an estimate probability of the positive class.

Following the above, the probability of the other class is therefore defined as simply as below:

$$ 1 - a $$

thus:

$$ P(y = 1) = 0.87 $$

and in turn:

$$ P(y = 1) = 0.13 $$

From this example, sigmoid is proven not to be completely obsolete. Instead, Sigmoid is most useful for particular output-based tasks, especially binary probabilities. Sigmoid should be avoided for tasks where it would function as the default hidden-layer activation function.

## Regression Outputs:
Not every neural network performs classification as we aim to here.

Suppose a network is trying to determine, or predict, temperature:

$$ \hat{y} = 21.7 $$

There is no reason for this output to be restricted to a range such as:

$$ [0,1] $$

or even:

$$ [-1, 1] $$

In any typical regression problem, the final output may in turn simply be just the raw value, i.e. the output defined such as below:

$$ \hat{y} = z $$

This is often referred to as a linear activation, or identity activation, as it directly reflections the functionality of the Identity function, as shown below:

$$ f(x) = x $$

The main takeaway from the above is that the output layer's activation depends heavily on what the network is aiming to represent or predict.

## How to Choose an Activation Function
There is often no single activation function that is correct to use, or in other words, universally appropriate. A rough guideline is given below:

| Value taken by Activation Function | Activation Function Choice |
|------------------------------------|----------------------------|
| Hidden Layer | ReLU, or a ReLU variant |
| Binary Output | Sigmoid |
| Mutliclass Output | Softmax |
| Regression Output | Identity |

These are very very rough conventions, and should not be interpreted through reading of this as immutable rules. A much wider range of functions are often used in heavily designed systems.

## Activation Functions & Backpropagation
Activation functions become particularly important when backpropagation occurs. Unlike forward propagation, backpropagation does not simply calculate a value and apply an activation function. Rather, backpropagation heavily relies on determining how changes in the value of `z` per layer in turn affect the value of `a`.

This requires a derivative operation, as shown:

$$ \frac{da}{dz} = f'(z) $$

which is why knowing the derivative of any relevant activation function is so vital.

For example, ReLU's derivative is:

$$ 
f'(z) =
\begin{cases}
0 & z < 0 \\
1 & z > 0
\end{cases}
$$

and Sigmoid's derivative:

$$ f'(z) = \sigma (z)(1-\sigma (z))

and tanh's derivative:

$$ f'(z) = 1 - tanh^{2}(z) $$

Where these derivatives become part of the chain of derivatives used to calculate gradients during backpropagation. In turn, this highlights the significance of the activation functions' mathematical properties, which directly impact how easily a network can be trained.

## Sources:
ill come back to this one. sorry.




