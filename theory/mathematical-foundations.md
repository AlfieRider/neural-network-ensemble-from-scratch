# Mathematical Foundations:
Neural networks cannot be described without vital mathematical notation, involving scalars, vectors, matrices, etc. Therefore, it would only make sense to explain the concepts required to understand (and in turn, to implement) these neural networks here.

## Scalars:
A scalar is simply a single numerical value.

Example: `x = 5`
> Here, x itself is a scalar, as it contains one number.

Scalars are effectively the simplest type of object used in neural networks (mathematically).

The bias of a neuron is a scalar value. Individual weights and input values are also scalars.

Python example 
```
weight = 0.5
```

## Vectors:
A vector is an ordered collection of numbers.

Example: `x = [2, 3, 5]`
> This vector contains values 2, 3, and 5.
> X here is a vector.

Python initial example:
```
x = [2, 3, 5]
```
It is vital to appreciate that this is only a starting point for conceptualisation purposes.

### What does a vector represent?
A vector is useful whenever several related numerical values need to be treated as a single object.

Example: Describing a person through their `height`, `weight`, and `age`.

$$
\mathbf{x} =
\begin{bmatrix}
180 \\
50 \\
20
\end{bmatrix}
$$
> Where height = 180, weight = 50, and age = 20.
> The entire collection is the vector x.

In a neural network, an input example will often be represented as a vector of numerical features.

For an image classifier, the elements may instead represent pixel values.

## Vector Notation:
There are many ways to write vectors.

Column Vector:

$$
\mathbf{x} =
\begin{bmatrix}
2 \\
3 \\
5
\end{bmatrix}
$$

Parentheses Column Vector:

$$
\mathbf{x} =
\begin{pmatrix}
2 \\
3 \\
5
\end{pmatrix}
$$

When layout isn't important:

$$
\mathbf{x} = (2,3,5)
$$

### Vector Elements:
The elements (components also) of a vector are the individual values inside it.

A vector `x` containing `n` elements:

$$
\mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

## Vector Shape:
The number of elements within a vector, often referred to as its size, length, or dimension.

Example 1:

$$
\mathbf{x} =
\begin{bmatrix}
2 \\
3 \\
5
\end{bmatrix}
$$

Example 1 represents a `3-dimensional` column vector. Its shape can be described as `3 * 1`, as it has 3 rows in one column.

$$
(3,1)
$$

The above is another way of describing this vector's shape; 3 rows, 1 column.

## Row Vectors & Column Vectors:
The distinction of row and column vectors are vital.

Column Vector:

$$
\mathbf{x} =
\begin{bmatrix}
2 \\
3 \\
5
\end{bmatrix}
$$

Has shape:

$$
3 \times 1
$$

$$
(3,1)
$$

Row Vector:

$$
\mathbf{x} =
\begin{bmatrix}
2 & 3 & 5
\end{bmatrix}
$$

Has shape:

$$
1 \times 3
$$

$$
(1,3)
$$

These two examples contain identical values. However, the different orientation is a vital difference considered in performing matrix multiplication.

## Transpose
The transpose of a vector (or matrix) switches its rows and columns. The transpose is represented by superscript `T`.

Example:

$$
\mathbf{x} =
\begin{bmatrix}
2 \\
3 \\
5
\end{bmatrix}
$$

has transpose:

$$
\mathbf{x} =
\begin{bmatrix}
2 & 3 & 5
\end{bmatrix}
$$

Thus the shape of the vector changes from:

$$
(3,1)
$$

to:

$$
(1,3)
$$

## Vector Addition:
Vectors with the same element number can be added together.

Example:

$$
\mathbf{a} =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

and:

$$
\mathbf{b} =
\begin{bmatrix}
4 \\
5 \\
6
\end{bmatrix}
$$

Can be added as shown:

$$
\mathbf{c} =
\begin{bmatrix}
1+4 \\
2+5 \\
3+6
\end{bmatrix}
$$

producing:

$$
\mathbf{c} =
\begin{bmatrix}
5 \\
7 \\
9
\end{bmatrix}
$$

Python example:
```
import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])

c = a + b
```
The above code outputs `[5,7,9]`.

## Scalar Multiplication:
A vector can be multiplied by a scalar.

Example:

$$
\mathbf{x} =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

multiplied by:

$$
a = 2
$$

This occurs such that as:

$$
2
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix} =
\begin{bmatrix}
2 \\
4 \\
6
\end{bmatrix}
$$

Python example:
```
import numpy as np

x = np.array([1,2,3])
result = 2 * x
```
Where `result = [2,4,6]`

## Dot Products:
One of the most vital operations for neural networks.

Consider two vectors:

$$
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix} ,
\begin{bmatrix}
4 \\
5 \\
6
\end{bmatrix}
$$

The dot product of these vectors is calculated via multiplying corresponding elements, then adding the results.

Therefore:

$$
(1)(4) + (2)(5) + (3)(6)
$$
$$
= 4 + 10 + 18
$$
$$
= 32
$$

As shown, the result produced is a single scalar, here being `32`.

## Dot Product Notation:
There are several forms of notation for the dot product.

One:

$$
\mathbf{a}\cdot\mathbf{b}
$$
> a and b are vectors here.

A common alternative:

$$
\mathbf{a}^T\mathbf{b}
$$

> See that this alternative notation involves transpose.

Take the following column vectors:

$$
\mathbf{a} = 
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix} ,
\mathbf{b} =
\begin{bmatrix}
4 \\
5 \\
6
\end{bmatrix}
$$

Transposing `a` produces:

$$
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
$$

Then the following:

$$
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
\begin{bmatrix}
4 \\
5 \\
6
\end{bmatrix}
$$

produces:

$$
(1)(4) + (2)(5) + (3)(6)
$$
$$
= 32
$$

Therefore:

$$
\boxed{\mathbf{a}^T\mathbf{b}=32}
$$

Expect this notation to constantly occur when describing individual neurons.

## Dot Products and Neurons:
The dot product is particularly useful, as a neuron essentially performs a weighted sum of its inputs.

Suppose a neuron receives three inputs:

$$
\mathbf{x} =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

which has three corresponding weights:

$$
\mathbf{w} =
\begin{bmatrix}
0.5 \\
0.2 \\
-0.1
\end{bmatrix}
$$

The weighted sum is expressed as:

$$
(0.5)(1) + (0.2)(2) + (-0.1)(3)
$$

Instead of explicitly writing all of this, the dot product notation can be used instead:

$$
\mathbf{W}^T\mathbf{x}
$$

Therefore:

$$
(0.5)(1) + (0.2)(2) + (-0.1)(3)
$$
$$
= 0.5 + 0.4 - 0.3
$$
$$
= 0.6
$$

If the neuron has a bias, `b`, such that:

$$
b = 0.1
$$

then:

$$
z = \mathbf{W}^T\mathbf{x} + b
$$

becomes:

$$
z = 0.6 + 0.1
$$
$$
= 0.7
$$

Note that this is logically equivalent to writing:

$$
z = w_1x_1 + w_2x_2 + w_3x_3 + b
$$

Vector notation simply allows more compact expression of this.

## Matrices:
A rectangular grid/array/arrangement of numbers, consisting of both rows and columns.

Example:

$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

therefore is of shape:

$$
2 \times 3
$$
$$
(2,3)
$$

Matrices are most useful when multiple vectors/sets of values need to be worked on simultaneously.

Python example:

```
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
  ])
```

## Matrix Notation:
Matrices are commonly represented with use of capital letters.

Example:

$$
W =
\begin{bmatrix}
0.5 & 0.3 & -0.1 \\
0.3 & -0.4 & 0.7
\end{bmatrix}
$$

Vectors are often represented through (bold) lowercase letters:

$$
\mathbf{x}
$$

whereas matrices conventionally use uppercase letters:

$$
W
$$

An example of formal use of this notation:

$$
\mathbf{z} = W\mathbf{x}+\mathbf{b}
$$
> vector Z = matrix W * vector x + vector b

## Matrix Dimensions and Shapes:
Matrix multiplication (explained later) only works when the dimensions of matrices are compatible.

Consider:

$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

The shape is described as shown:

$$
2 \text{ rows} \times 3 \text{ columns}
$$
$$
2 \times3
$$
$$
(2,3)
$$

Now consider:

$$
\mathbf{x} =
\begin{bmatrix}
7 \\
8 \\
9
\end{bmatrix}
$$

This has shape described as below:

$$
3 \text{ rows} \times 1 \text{ column}
$$
$$
3 \times1
$$
$$
(3,1)
$$

Therefore, the multiplication of these two can occur, as the inner dimensions match:

$$
A\mathbf{x}
$$

$$
= \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
\begin{bmatrix}
7 \\
8 \\
9
\end{bmatrix}
$$

The resulting shape is therefore:

$$
=> (2\times\mathbf{3})(\mathbf{3}\times1)
$$

$$
2\times1
$$
> As the inner 3s match, the resulting shape of a matrix multiplication would be dictated by the outer two values, being (2,1)

## Matrix-Vector Multiplication:
Learnt best through an example, the same as above.

Example cont.

$$
= \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
\begin{bmatrix}
7 \\
8 \\
9
\end{bmatrix}
$$

The first row is multiplied by the vector:

$$
(1)(7)+(2)(8)+(3)(9)
$$
$$
= 7+16+27
$$
$$
= 50
$$

The second row is also then multiplied by the vector:

$$
(4)(7)+(5)(8)+(6)(9)
$$
$$
= 28+40+54
$$
$$
= 122
$$

Thus the result is as shown:

$$
\begin{bmatrix}
50 \\
122
\end{bmatrix}
$$

In other words, each row of the matrix performs a dot product with the vector.

## Matrix Multiplication and Neural Network Layers:
Suppose a neural network layer contains:
- 3 inputs values
- 2 neurons
Each neuron should therefore have (need) 3 weights.

The first neuron may have:

$$
\mathbf{w_1}=
\begin{bmatrix}
w_{11} \\
w_{12} \\
w_{13}
\end{bmatrix}
$$

The second neuron may have

$$
\mathbf{w_2}=
\begin{bmatrix}
w_{21} \\
w_{22} \\
w_{23}
\end{bmatrix}
$$

Instead of storing these separately, these can be stored in a single matrix, as shown:

$$
W =
\begin{bmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23}
\end{bmatrix}
$$

This has shape:

$$
2\times3
$$
> as there are 2 neurons, and 3 inputs per neuron.

Now, let:

$$
\mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
$$

The entire layer's weighted sums can be calculated with:

$$
W\mathbf{x}
$$

which gives:

$$
\begin{bmatrix}
w_{11}x_1 + w_{12}x_2 + w_{13}x_3 \\
w_{21}x_1 + w_{22}x_2 + w_{23}x_3
\end{bmatrix}
$$

The first element is the weighted sum for neuron 1, and the second element is the weighted sum for neuron 2.

In other words, one matrix multiplication has performed the weighted sum calculation for every neuron in the layer effectively simultaneously.

## Bias vectors
When a layer contains multiple neurons, each neuron generally has its own bias. Therefore, instead of having a single scalar bias, `b`, a vector bias may exist instead:

$$
\mathbf{b} =
\begin{bmatrix}
b_1 \\
b_2 \\
\end{bmatrix}
$$

> For two given neurons, there are two biases.

The complete calculation for the layer is in turn:

$$
\mathbf{z} = W\mathbf{x} + \mathbf{b}
$$

Example:

$$
W =
\begin{bmatrix}
0.5 & 0.2 & -0.1 \\
0.3 & -0.4 & 0.7
\end{bmatrix} ,
\mathbf{x} =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

and:

$$
\mathbf{b} =
\begin{bmatrix}
0.1 \\
0.2
\end{bmatrix}
$$

First calculate:

$$
W\mathbf{x}
$$
> the weighted sum for each neuron.

Then add the corresponding bias to each result.
> as shown above.

The result is:

$$
\mathbf{z} =
\begin{bmatrix}
z_1 \\
z_2
\end{bmatrix}
$$

where z₁ belongs to the first neuron, and z₂ the second.

## General Idea of Matrix Multiplication:
The principle of matrix multiplication has been described in full above. See the below as a summary thus far.

For:

$$
A\mathbf{x}
$$

each row of A is used to calculate the dot product with `x`.

Given two general matrices,

$$
A_{m\times n}
$$
$$
B_{n\times p}
$$

The multiplication:

$$
AB
$$

produces a matrix with shape:

$$
m\times p
$$

The important underlying rule here is that the inner dimensions must match.

Matching example:

$$
(2\times3)(3\times1) => (2\times1)
$$
> valid, as the inner 3s match.

Invalid example:

$$
(2\times3)(1\times2) => \text{invalid}
$$
> invalid, as the inner dimensions of 3 and 1 do not match.

This is also useful when checking any neural network equations.

## Python - Mathematics Display:
The vector shown below:

$$
\mathbf{x}
\begin{bmatrix}
1 //
2 //
3
\end{bmatrix}
$$

can be represented as:

```
import numpy as np

x = np.array([1,2,3])
```

The matrix shown below:

$$
W =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

can be represented as:

```
import numpy as np

W = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

The Matrix-vector multiplication:

$$
W\mathbf{x}
$$

can be performed as shown:

```
import numpy as np
#x and W declared as above

result = W @ x
```
> The python @ operator represents matrix multiplication
> For numpy arrays, W * x is element-wise multiplication when the shape permits it.
> For numpy arrays W @ x performs matrix multiplication itself.

## A complete Neural Network example:
A summary-like combination of all that is explained above.

Suppose a layer contains:
- 3 inputs
- 2 neurons

The input is:

$$
\mathbf{x} =
\begin{bmatrix}
2 \\
3 \\
5
\end{bmatrix}
$$

The weight matrix is:

$$
W =
\begin{bmatrix}
0.5 & 0.2 & -0.1 \\
0.3 & -0.4 & 0.7
\end{bmatrix}
$$

The biases are:

$$
\mathbf{b} =
\begin{bmatrix}
0.1 \\
0.2
\end{bmatrix}
$$

The first layer calculates the following:

$$
W\mathbf{x}
$$

$$
= \begin{bmatrix}
(0.5)(2)+(0.2)(3)+(-0.1)(5) \\
(0.3)(2)+(-0.4)(3) + (0.7)(5)
\end{bmatrix}
$$
$$
= \begin{bmatrix}
1.1 \\
2.9
\end{bmatrix}
$$

The bias is now added, as shown:

$$
\begin{bmatrix}
1.1 \\
2.9
\end{bmatrix} +
\begin{bmatrix}
0.1 \\
0.2
\end{bmatrix}
$$
$$
= \begin{bmatrix}
1.2 \\
3.1
\end{bmatrix}
$$

This final vector holds the pre-activation values for the two neurons given.

Following this, an activation function is then applied to the values. For this example, ReLU will be used as a substitue:

$$
\mathop{\text{ReLU}}(z)=\max(0,z)
$$

then gives:

$$
\begin{bmatrix}
\mathop{\text{ReLU}}(1.2) \\
\mathop{\text{ReLU}}(3.1)
\end{bmatrix}
$$

in turn giving:

$$
\begin{bmatrix}
1.2 \\
3.1
\end{bmatrix}
$$

Therefore, the entire layer can then be expressed compactly, as shown below:

$$
\boxed{\mathbf{a}=\mathop{\text{ReLU}}(W\mathbf{x}+\mathbf{b})}
$$

Take mental note of this equation; this is extremely important for forward propagation.

## Why bother with this notation?
This section serves as a sort of summary to the above.

At first, an expression such as shown below may feel undesirably excessive:

$$
\mathbf{a}=\mathop{\text{ReLU}}(W\mathbf{x}+\mathbf{b})
$$

especially in comparison to such of:

$$
a = \mathop{\text{ReLU}}(W @ x + b)
$$

However, nothing too special is really happening in terms of complexity. The mathematical process of equation 1 of this section is written below:
1. Take the input vector `x`
2. Multiply `x` by weight matrix `W`
3. Add the bias vector `b` to ^^
4. Apply `ReLU` to each resulting value (can substitute any activiation function)
5. Obtain the output vector `a`

The given equation works, and becomes very valuable because of the fact, such for whether a layer contains two neurons, or two million neurons. Additionally, its nice and simplistic to describe the entire layer at once.

## Notation Summary:
| Notation | Meaning |
|----------|---------|
| x | A scalar |
| **x** | A vector |
| xᵢ | i-th vector element |
| W | a martix (e.g. Weight Matrix) |
| **b** | a bias vector |
| xᵀ | transpose of a vector |
| **a** + **b** | Vector addition |
| x*y | scalar multiplication |
| **a**ᵀb | dot product |
| Wx | Matrix-vector multiplication |
| Wx + b | Weighted sums + bias for a layer |
| ReLU(x) | ReLU applied element-wise |

## What to read next?
The idea behind this specific markdown file was to prepare myself for the foundational mathematics behind forward propagation. Thus, naturally, I highly recommend that any reader reads `forward-propagation.md` next! 

Do note that this information will not be repeated again, so either learn this well or have it ready to go in another tab if needbe!

## Sources:
https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/1.-vectors-and-matrices

https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/vectors

https://www.khanacademy.org/math/multivariable-calculus/thinking-about-multivariable-function/x786f2022:vectors-and-matrices/a/matrices--intro-mvc

https://openstax.org/books/algebra-and-trigonometry-2e/pages/10-8-vectors

https://openstax.org/books/calculus-volume-3/pages/2-3-the-dot-product










