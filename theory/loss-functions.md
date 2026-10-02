# Loss Functions
A neural network produces a prediction via the forward propagation process. For example, for a handwritten-digit image-classifier tool, the network may produce the following:

$$
\hat{y} =
\begin{bmatrix}
0.01 \\
0.02 \\
0.03 \\
0.04 \\
0.85 \\
0.02 \\
0.01 \\
0.01 \\
0.01 \\
0.00
\end{bmatrix}
$$
> in this case, the network effectively determines the input image to be a 4 (digit).

However, this cannot be guaranteed to be a remotely good prediction; a network may be just as likely to output a 99% prediction of a 3 when the input was a 1. Therefore, it is incredibly helpful to have an actually mathematical way of comparing the network's prediction with the correct answer. This is what a `loss function` aims to do.

A loss function will take a prediction (the network's output), and then take the corresponding target prediction (i.e. what it should be), to then produce a numerical value representing how poorly the model performed on that example. Conceptually speaking, it is shown as below:

$$ (prediction, target) -> loss $$

During training, the network then attempts to adjust its parameters via this such that the loss becomes smaller.

## What is a Loss Function?
This may best be explained through a practical example. Suppose the true value, `y`, is given below:

$$ y = 10 $$

and the network in turn predicts:

$$ \hat{y} = 8 $$

This prediction is, well, clearly wrong... A loss function takes these values and outputs how 'clearly' wrong this clearly is.

An example of one loss function, used for this case, is given below as:

$$ 
