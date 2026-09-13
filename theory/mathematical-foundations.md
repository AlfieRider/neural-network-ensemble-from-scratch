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

Example: `f(x) = [2, 3, 5]`
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
...
