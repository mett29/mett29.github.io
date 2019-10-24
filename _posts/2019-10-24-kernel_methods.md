---
title: 'Kernel Methods'
date: 2019-10-24
permalink: /posts/2019/10/kernel_methods/
usemathjax: true
tags:
  - kernel methods
  - radial basis function
  - gaussian process
  - kernel trick
---

In this post we will talk about **Kernel Methods**, explaining the math behind them in order to understand how powerful they are and for what tasks they can be used in an efficient way.

**Disclaimer:** *the following notes were written following the slides provided by the professor Restelli at Polytechnic of Milan and the book '[Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732)'.*

# Kernel Methods

Kernel methods are non-parametric and memory-based (e.g. K-NN), i.e. methods that involve storing the entire training set in order to make predictions for future data points, that typically require a metric to be defined that measures the similarity of any two vectors in input space, and are generally fast to ‘train’ but slow at making predictions for test data points.

Many linear parametric models can be re-cast into an equivalent ‘dual representation’ in which the predictions are also based on linear combinations of a **kernel function** evaluated at the training data points. 
As we shall see, for models which are based on a fixed nonlinear feature space mapping $\phi(\boldsymbol{x})$, the kernel function is given by the relation

$k(\boldsymbol{x},\boldsymbol{x'}) = \phi(\boldsymbol{x})^T\phi(\boldsymbol{x'})$

Note that the kernel is a symmetric function of its argument, so that $k(\boldsymbol{x},\boldsymbol{x'}) = k(\boldsymbol{x'},\boldsymbol{x})$ and it can be interpreted as similarity between $\boldsymbol{x}$ and $\boldsymbol{x'}$.

The simplest example of a kernel is obtained by considering the identity mapping for the feature space, so that $\phi(\boldsymbol{x}) = \boldsymbol{x}$ (we are not transforming the features' space), i.e. $k(\boldsymbol{x},\boldsymbol{x'}) = \boldsymbol{x}^T\boldsymbol{x'}$, called **linear kernel**.

The concept of a kernel formulated as an inner product in a feature space allows us to build interesting extensions of many well-known algorithms by making use of the **kernel trick**, also known as kernel substitution. 
The general idea is that if we have an algorithm formulated in such a way that the input vector $\boldsymbol{x}$ enters only in the form of *scalar products*, then we can replace that scalar product with some other choice of kernel.

There exist various form of kernels functions:

- $k(\boldsymbol{x},\boldsymbol{x'}) = k(\boldsymbol{x}-\boldsymbol{x'})$, called stationary, because they are invariant to translations in input space.
- $k(\boldsymbol{x},\boldsymbol{x'}) = k(\|\|\boldsymbol{x}-\boldsymbol{x'}\|\|)$, called homogeneous kernels and also known as **radial basis functions**, which depend only on the magnitude of the distance (typically Euclidean) between the arguments.
- etc.

## Dual representation

Consider a linear regression model in which the parameters are obtained by minimizing the regularized sum-of-squares error function

$L_{\boldsymbol{w}} = \frac{1}{2}\sum_{n=1}^{N}(\boldsymbol{w}^T\phi(\boldsymbol{x_n})-t_n)^2 + \frac{\lambda}{2}\boldsymbol{w}^t\boldsymbol{w}$

What we want is to make $\boldsymbol{w}$ and $\phi$ disappear.
Setting the gradient of $L_{\boldsymbol{w}}$ w.r.t. $\boldsymbol{w}$ equal to zero we obtain

$\boldsymbol{w} = -\frac{1}{\lambda}\sum_{n=1}^{N}(\boldsymbol{w}^T\phi(\boldsymbol{x_n})-t_n)\phi(\boldsymbol{x_n}) = \sum_{n=1}^{N}a_n\phi(\boldsymbol{x_n}) = \Phi^T\boldsymbol{a}$

where $\Phi$ is the usual design matrix and $a_n = -\frac{1}{\lambda}(\boldsymbol{w}^T\phi(\boldsymbol{x_n})-t_n)$.

We now define the **Gram matrix** $K = \phi \times \phi^T$ an $N \times N$ symmetric matrix, with elements

$K_{nm} = \phi(\boldsymbol{x_n})^T\phi(\boldsymbol{x_m}) = k(\boldsymbol{x_n},\boldsymbol{x_m})$

Given $N$ vectors, the Gram matrix is the matrix of all inner products, hence for example if we take the first row and the first column we will find the kernel between $\boldsymbol{x_1}$ and $\boldsymbol{x_1}$.

Substituting $\boldsymbol{w} = \Phi^T\boldsymbol{a}$ into $L_{\boldsymbol{w}}$ gives

$L_{\boldsymbol{w}} = \frac{1}{2}\boldsymbol{a}^T\Phi\Phi^T\Phi\Phi^T\boldsymbol{a} - \boldsymbol{a}^T\Phi\Phi^T\boldsymbol{t} + \frac{1}{2}\boldsymbol{t}^T\boldsymbol{t} + \frac{\lambda}{2}\boldsymbol{a}^t\Phi\Phi^T\boldsymbol{a}$

where $\boldsymbol{t} = (t_1,...,t_N)^T$.

In terms of the Gram matrix, the sum-of-squares error function can be written as

$L_{\boldsymbol{a}} = \frac{1}{2}\boldsymbol{a}^TKK\boldsymbol{a} - \boldsymbol{a}^TK\boldsymbol{t} + \frac{1}{2}\boldsymbol{t}^T\boldsymbol{t} + \frac{\lambda}{2}\boldsymbol{a}^tK\boldsymbol{a}$

And solving for $\boldsymbol{a}$

$\boldsymbol{a} = (K + \lambda\boldsymbol{I_N})^{-1}\boldsymbol{t}$

If we substitute this back into the linear regression model, we obtain the following prediction for a new input $\boldsymbol{x}$

$y(\boldsymbol{x}) = \boldsymbol{w}^T\phi(\boldsymbol{x}) = a^T\Phi\phi(\boldsymbol{x}) = \boldsymbol{k}(\boldsymbol{x})^T(K+\lambda\boldsymbol{I_N})^{-1}\boldsymbol{t}$

where $\boldsymbol{k}(\boldsymbol{x})$ has elements $k_n(\boldsymbol{x}) = k(\boldsymbol{x_n},\boldsymbol{x})$, that means how much each sample is similar to the query vector $\boldsymbol{x}$.

Thus we see that the dual formulation allows the solution to the least-squares problem to be expressed entirely in terms of the kernel function $k(\boldsymbol{x},\boldsymbol{x'})$.
In this new formulation, we determine the parameter vector a by inverting an $N \times N$ matrix, whereas in the original parameter space formulation we had to invert an $M \times M$ matrix in order to determine $\boldsymbol{w}$. 
Because $N$ is typically much larger than $M$, the dual formulation does not seem to be particularly useful. However, the advantage of the dual formulation, as we shall see, is that **it is expressed entirely in terms of the kernel function** $k(\boldsymbol{x},\boldsymbol{x'})$. We can therefore work directly in terms of kernels and avoid the explicit introduction of the feature vector $\phi(\boldsymbol{x})$, which allows us implicitly to use feature spaces of high, even infinite, dimensionality.

## Constructing kernels

In order to exploit kernel substitution, we need to be able to construct valid kernel functions.

*First method*

One approach is to choose a feature space mapping $\phi(\boldsymbol{x})$ and then use this to find the corresponding kernel.
In case of one-dimensional input space:

$k(\boldsymbol{x},\boldsymbol{x'}) = \phi(\boldsymbol{x})^T\phi(\boldsymbol{x}') = \sum_{i=1}^{M}\phi_i(\boldsymbol{x})\phi_i(\boldsymbol{x'})$

where $\phi_i(\boldsymbol{x})$ are the basis functions.

*Second method*

An alternative approach is to construct kernel functions directly. In this case, we must ensure that the function we choose is a valid kernel, in other words that it corresponds to a scalar product in some (perhaps infinite dimensional) feature space.

For example, consider the kernel function $k(\boldsymbol{x},\boldsymbol{z}) = (\boldsymbol{x}^T\boldsymbol{z})^2$ in two dimensional space:

$k(\boldsymbol{x},\boldsymbol{z}) = (\boldsymbol{x}^T\boldsymbol{z})^2 = (x_1z_1+x_2z_2)^2 = x_1^2z_1^2 + 2x_1z_1x_2z_2 + x_2^2z_2^2 = (x_1^2,\sqrt{2}x_1x_2,x_2^2)(z_1^2,\sqrt{2}z_1z_2,z_2^2)^T = \phi(\boldsymbol{x})^T\phi(\boldsymbol{z})$

More generally, however, we need a simple way to test whether a function constitutes a valid kernel without having to construct the function $\phi(\boldsymbol{x})$ explicitly, and fortunately there is a way.

A necessary and sufficient condition for a function $k(\boldsymbol{x},\boldsymbol{x'})$ to be a valid kernel is that the Gram matrix $K$ is positive semidefinite for all possible choices of the set $\{\boldsymbol{x_n}\}$.

One powerful technique for constructing new kernels is to build them out of simpler kernels as building blocks.
Given valid kernels $k_1(\boldsymbol{x},\boldsymbol{x'})$ and $k_2(\boldsymbol{x},\boldsymbol{x'})$, the following new kernels will also be valid:

1. $k(\boldsymbol{x},\boldsymbol{x'}) = ck_1(\boldsymbol{x},\boldsymbol{x'})$
2. $k(\boldsymbol{x},\boldsymbol{x'}) = f(\boldsymbol{x})k_1(\boldsymbol{x},\boldsymbol{x'})f(\boldsymbol{x})$
3. $k(\boldsymbol{x},\boldsymbol{x'}) = q(k_1(\boldsymbol{x},\boldsymbol{x'}))$, where $q()$ is a polynomial with non-negative coefficients.
4. $k(\boldsymbol{x},\boldsymbol{x'}) = e^{k_1(\boldsymbol{x},\boldsymbol{x'})}$
5. $k(\boldsymbol{x},\boldsymbol{x'}) = k_1(\boldsymbol{x},\boldsymbol{x'}) + k_2(\boldsymbol{x},\boldsymbol{x'})$
6. $k(\boldsymbol{x},\boldsymbol{x'}) = k_1(\boldsymbol{x},\boldsymbol{x'})k_2(\boldsymbol{x},\boldsymbol{x'})$
7. $k(\boldsymbol{x},\boldsymbol{x'}) = k_3(\phi(\boldsymbol{x}),\phi(\boldsymbol{x'}))$, where $\phi(\boldsymbol{x})$ is a function from $\boldsymbol{x}$ to $\mathcal{R}^M$.
8. $k(\boldsymbol{x},\boldsymbol{x'}) = \boldsymbol{x}^TA\boldsymbol{x'}$, where $A$ is a symmetric positive semidefinite matrix.
9. $k(\boldsymbol{x},\boldsymbol{x'}) = k_a(x_a,x'_a) + k_b(x_b,x'_b)$, where $x_a$ and $x_b$ are variables with $\boldsymbol{x} = (x_a,x_b)$ and $k_a$ and $k_b$ are valid kernel functions.
10. $k(\boldsymbol{x},\boldsymbol{x'}) =k_a(x_a,x'_a)k_b(x_b,x'_b)$

A commonly used kernel is the **Gaussian kernel**:

$$k(\boldsymbol{x},\boldsymbol{x'}) = e^{-\frac{||\boldsymbol{x}-\boldsymbol{x'}||^2}{2\sigma^2}}$$

where $\sigma^2$ indicates how much you generalize, so $underfitting \implies reduce \ \sigma^2$.

Lastly, there is another powerful approach, which makes use of probabilistic generative models, allowing us to apply generative models in a discriminative setting. 
Generative models can deal naturally with missing data and in the case of hidden Markov models can handle sequences of varying length. By contrast, discriminative models generally give better performance on discriminative tasks than generative models. It is therefore of some interest to combine these two approaches. One way to combine them is to use a generative model to define a kernel, and then use this kernel in a discriminative approach.

Given a generative model $p(\boldsymbol{x})$ we can define a kernel by

$k(\boldsymbol{x},\boldsymbol{x'}) = p(\boldsymbol{x})p(\boldsymbol{x'})$

This is clearly a valid kernel function and it says that two inputs $\boldsymbol{x}$ and $\boldsymbol{x'}$ are similar if they both have high probabilities.

## Radial Basis Functions

A radial basis function, RBF,  $\phi(\boldsymbol{x})$ is a function with respect to the origin or a certain point $c$, i.e. $\phi(\boldsymbol{x}) = f(\|\|\boldsymbol{x}-\boldsymbol{c}\|\|)$, where typically the norm is the standard Euclidean norm of the input vector, but technically speaking one can use any other norm as well.

The RBF learning model assumes that the dataset  $\mathcal{D} = (x_n,y_n), n=1,...,N$ influences the hypothesis set  $h(x)$, for a new observation $x$, in the following way:

$$
h(x) = \sum_{n=1}^{N}w_n e^{-\gamma ||x-x_n||^2}
$$

which means that each $x_i$ of the dataset influences the observation in a gaussian shape. Of course, if a datapoint is far away from the observation its influence is residual (the exponential decay of the tails of the gaussian make it so). It is an example of a localized function ($x \rightarrow \infty \implies \phi(x) \rightarrow 0$).

Ok, so, given this type of basis function, how do we find $\boldsymbol{w}$?

The choice of $\boldsymbol{w}$ should follow the goal of minimizing the in-sample error of the dataset $\mathcal{D}$:

$\sum_{m=1}^{N}w_m e^{-\gamma \|\|x_n-x_m\|\|^2} = y_n$ for each datapoint $x_n \in \mathcal{D}$

which in matrix form can be expressed as

$\Phi\boldsymbol{w} = \boldsymbol{y}$

$\boldsymbol{w} = \Phi^{-1}\boldsymbol{y}$

Note that $\Phi$ is not a square matrix, so we have to compute the pseudo-inverse:

$\boldsymbol{w} = (\Phi^T\Phi)^{-1}\Phi^T\boldsymbol{y}$ (recall what we saw in the Linear Regression chapter)

Of course, all this can be adapted for classification problems:

$$h(x) = sign(b + \sum_{n=1}^{N}w_n e^{-\gamma ||x-x_n||^2})$$

In machine learning, radial basis functions are most commonly used as a kernel for classification with the support vector machine (SVM).

## Gaussian Processes

In probability theory and statistics, a Gaussian process is a stochastic process (a collection of random variables indexed by time or space), such that every finite collection of those random variables has a multivariate normal distribution, i.e. every finite linear combination of them is normally distributed. The distribution of a Gaussian process is the joint distribution of all those (infinitely many) random variables, and as such, it is a distribution over functions with a continuous domain, e.g. time or space.

A machine-learning algorithm that involves a Gaussian process uses lazy learning and a measure of the similarity between points (the kernel function) to predict the value for an unseen point from training data. The prediction is not just an estimate for that point, but also has uncertainty information—it is a one-dimensional Gaussian distribution.

I will not enter in the details, for which I direct you to the book [Pattern Recognition and Machine Learning]([https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)), but the idea is that Gaussian Process approach differs from the Bayesian one thanks to the non-parametric property. Indeed, it finds a distribution over the possible **functions**  $f(x)$ that are consistent with the observed data.

More precisely, taken from the textbook [Machine Learning: A Probabilistic Perspective](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020/):

*A GP defines a prior over functions, which can be converted into a posterior over functions once we have seen some data. Although it might seem difficult to represent a distrubution over a function, it turns out that we only need to be able to define a distribution over the function’s values at a finite, but arbitrary, set of points, say $x_1,...,x_N$. A GP assumes that $p(f(x_1),...,f(x_N))$ is jointly Gaussian, with some mean $\mu(x)$ and covariance $\sum (x)$ given by $\sum_{ij} = k(x_i,x_j)$, where $k$ is a positive definite kernel function. The key idea is that if $x_i$ and $x_j$ are deemed by the kernel to be similar, then we expect the output of the function at those points to be similar, too.*

In addition to the book, I highly recommend this post written by Yuge Shi: [Gaussian Process, not quite for dummies](https://yugeten.github.io/posts/2019/09/GP/)