---
title: 'Batch Normalization'
date: 2019-11-17
permalink: /posts/2019/11/batch_normalization/
usemathjax: true
tags:
  - batch normalization
---

In this post we will talk about **batch normalization**, explaining what it is and how it works!

**Disclaimer:** *These notes are for the most part a collection of concepts taken from the slides of the 'Artificial Neural Networks and Deep Learning' course at Polytechnic of Milan, the book 'Deep Learning' (Goodfellow-et-al-2016) and from some other online resources. I am simply putting together all the information to study for the exam and I thought it would be a good idea to upload them here since they can be useful for someone who is interested in this topic.*

# Batch Normalization

Batch normalization (Ioffe and Szegedy, 2015) is a method of adaptive reparametrization motivated by the difficulty of training very deep models.

You can find the paper here $\rightarrow$ [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

> Training Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift.
>
>> <cite>Ioffe and Szegedy, 2015, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift</cite>

Thus, in order to understand batch normalization, we first need to understand what is the **covariate shift** phenomenon.

## Covariate Shift

“Covariates” is just another name for the input “features”, often written as $X$. Covariate shift means the distribution of the features is different in different parts of the training/test data, breaking the i.i.d assumption.

More in general, in the whole field of data science this problem is very well known and it is called **dataset shift (or drifting)**. It occurs when the distribution of the training set and the test set is different, so no matter how well you trained your model, in the test set it will perform poorly. This problem is sometimes not mentioned, expecially in online competitions, because in that case the datasets are usually well organized and cleaned, having the same distribution in both the sets. This is not necessarily true in a real world scenario, where data might not have that level of quality. This is the case for example of finance, due to the always different market conditions. 

This is a good article about this problem: [Covariate Shift – Unearthing hidden problems in Real World Data Science](https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/)

Coming back to our neural networks, **internal covariate shift** refers to covariate shift occurring within a neural network, for example going from layer 2 to layer 3. This happens because, as the network learns and the weights are updated, the distribution of outputs of a specific layer in the network changes. This forces the higher layers to adapt to that drift, which slows down learning.

## Solution

Starting from the fact that it is known (*LeCun et al., 1998b; Wiesler & Ney, 2011*) that the network training converges faster if its inputs are whitened, the idea behind batch normalization is to apply the same procedure to the inputs of each layer.

{:refdef: style="text-align: center;"}
![algorithm](https://miro.medium.com/max/506/1*Hiq-rLFGDpESpr8QNsJ1jg.png)
{: refdef}

{:refdef: style="text-align: center;"}
[Image from the paper](https://arxiv.org/pdf/1502.03167.pdf)
{: refdef}

As we can see, we have two new parameters: $\gamma$ and $\beta$. The reason is that normalizing the mean and standard deviation of a unit can reduce the expressive power of the neural network containing that unit. To maintain this expressive power, it is common to replace $\hat{x_i}$ with $\gamma \hat{x_i} + \beta$.

The variables $\gamma$ and $\beta$ are learned parameters that allow the new variable to have any mean and standard deviation. Even if this can seem counterintuitive, the reason is that this new parametrization can represent the same family of functions of the input of the old one, but it has a different learning dynamics. More precisely, in the old parametrization the mean was determined by a complicated interaction between the parameters in the previous layers, while in the new one it is determined by the solely $\beta$, thus it is much easier to learn with gradient descent.

## Conclusions and observations

In practice batch normalization has shown to:

- improve gradient flow through the network
- allow higher learning rates
- reduce the strong dependence on initialization
- act as a form of regularization, reducing the need for dropout

Moreover, it makes possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes.

**However**, a recent paper questioned the reason for which batch normalization works, stating that 

>the real reason is that it makes the optimization landscape significantly smoother, inducing a more predictive and stable behavior of the gradients, allowing for faster training.
>
>> [How Does Batch Normalization Help Optimization?](https://arxiv.org/pdf/1805.11604.pdf)