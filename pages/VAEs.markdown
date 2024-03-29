---
layout: page
title:  "Variational Autoencoders"
date:   2022-02-01 21:52:57 +0100
permalink: /VAEs/
---

### **Introduction**
Variational Autoencoders (VAEs) are a class of generative neural networks that can encode data in a self-supervised manner. It consist of an encoder, a latent space (also called the bottleneck) and a decoder, where each latent variable in the latent space is assumed to have a probability distribution. They differ from a traditional autoencoder (AE) in that the latent space consists of stochastic variables, which need to be sampled. The sampling is carried out by using the reparameterization trick and variational inference, since evaluating the integral is intractable.  

**Content**
* [Intuition](#intuition)  
* [Model and Math](#model-and-math)  
* [Seminal Papers](#seminal-papers)  
* [Model Variations](#variations)  
* [Tricks of the Trade](#tricks-of-the-trade)  
* [Code Repositories](#code-repositories)  
&nbsp;

### **Intuition**<a name="#intuition"></a>
The VAE bears resemblance to the AE, but differ in some main aspects. The main goal of both models is to take an input and compress it into a latent space, using the encoder and then reconstruct the input again from the latent space through the decoder. This forces the encoder to extract prominent features of the input data, in order for the decoder to have sufficient information to reconstruct the input data.

In the VAE, the latent space have a probability distribution, often set to a standard normal distribution where each latent variable is independent of the others. Hence, the encoder is outputting a _distribution_, and not directly a latent variable as in the AE. The job of the encoder is more precisely to output a distribution of the latent variables, from which we can sample, that is _likely_ to produce good reconstructions from the decoder.

Since we have assumed a distribution for the latent space in the VAE, we can sample new examples by passing noise through the decoder, sampled with the same distribution as the latent space. This is what makes the VAE suitable to use as a generative model, once trained. However, in practice a vanilla VAE produce blurry samples of e.g. images, and other more sophisticated versions are necessary to generate good images.

As the job of the encoder is to effectively compress our input data, it can be utilized in semi-supervised learning. After training, the encoder can be finetuned on a smaller, labelled dataset. Or the VAE can be trained jointly on both labelled and unlabelled data at the same time (see [Variations](#variations)).

Another application of the VAE, is unsupervised clustering of data. While the model trains, the encoder and decoder tries to extract natural structure in the data, and organize similar data points in the vicinity of each other in the latent space. The latent space can then be visualized using projection methods such as PCA, t-SNE and UMAP, and clusters can be examined from here.  
&nbsp; 

### **Model and Math**<a name="#model-and-math"></a>
Two derivations of the Evidence Lower Bound (ELBO) exists, one from the ground up by Carl Doersch and one using Jensen's inequality. We first take the deriviation of Doersch, as it offers the most intuition.

**Derivation from Doersch**  
Overall we wish learn the distribution over our data $$P(X)$$, defined on some (high dimensional) space $$\chi$$, where $$X$$ could e.g. be images. We wish to represent the structure of our by a vector of latent variables $$z$$. This could be the components of handwritten digits, such that a set of latent variables could describe e.g. the digit 5 and the different ways to write a 5. Further, we wish to be able to easily sample the latent variables $$z$$ according to distribution $$P(z)$$ to produce new examples that are very similar to our original data X.

Formally, we have say that we have a vector of latent variables $$z$$ in a high-dimensional space $$\mathcal{Z}$$ which we can sample according to $$P(z)$$. We then have a family of deterministic functions $$f(z;\theta)$$, parameterized by a vector $$\theta$$ in some space $$\Theta$$ where $$f\,:\, \mathcal{Z} \times \Theta \rightarrow \chi$$. Here, $$f$$ will be deterministic, $$z$$ is random and $$\theta$$is fixed. We then have that $$f(z;\theta)$$ is a random variable in the space $$\chi$$. In our case, $$f$$ is a neural network (the **decoder**) and $$\theta$$ are the weights of the network that we wish to train. $$f(z;\theta)$$ (decoder) maps specific samples of $$z$$ into examples of e.g. images similar to our data distribution $$P(X)$$.

We wish to maximize the probability of each $$X$$ in the training set under the generative process, according to

$$\begin{equation}
P(X) = \int P(X|z;\theta)P(z)\, dz
\end{equation}$$

$$f(z;\theta)$$ has been replaced by a distribution $$P(X|z;\theta)$$, which make the dependence of $$z$$ on $$X$$ explicit. The choice of this output distribution is often chosen to be Gaussian: $$P(X|z;\theta) = \mathcal{N}(X|f(z;\theta), \sigma^2 \cdot I)$$. That is, it has mean $$f(z;\theta)$$ and a diagonal covariance matrix times some scalar $$\sigma^2$$. The prior $$P(z)$$ is often chosen to be a standard Gaussian: 
$$P(z) = \mathcal{N}(0, I)$$. In practice, this mean that we evaluate the generated $$X$$ on a normal distribution on how well it fit out data distribution $$P(X)$$.

Since it would quite ineffective to simply sample $$z$$ until we got good representations of P(x), we wish to have a function 
$$Q(z|X)$$ which can take a value of X and give us a distribution over over $$z$$ values that a likely to produce X. This is what will be our **encoder**. Given that $$Q(z|X)$$ does a good job of suggesting distributions over $$z$$ that are likely to produce X, the space will be much smaller and we can easier compute e.g. $$E_{z \sim Q} P(X\vert z)$$.  

In order for this to work, we need to relate $$E_{z \sim Q} P(X \vert z)$$ and $$P(x)$$. This next part will seem slightly convoluted a first, but will make sense in the end. Since $$Q(z)$$ is the distribution over $$z$$ that would yield good distributions to sample from, we stat with the Kullback-Liebler divergence (KL divergence or $$\mathcal{D}$$) between $$p(z \vert X)$$ and $$Q(z)

$$\begin{equation}
\mathcal{D}[Q(z) \| P(z \vert X)]=E_{z \sim Q}[\log Q(z)-\log P(z \vert X)]
\end{equation}$$

We get $$P(X)$$ and $$P(X \vert z)$$ into the equation by applying Bayes rule to $$P(z \vert X)$$

$$\begin{equation}
\mathcal{D}[Q(z) \| P(z \vert X)]=E_{z \sim Q}[\log Q(z)-\log P(X \vert z)-\log P(z)]+\log P(X)
\end{equation}$$

$$\log P(X)$$ comes out of the expectation as it does not depend on z. By negating both sides, rearranging and contracting part of $$E_{z\sim Q}$$ in the KL divergence, we get 

$$\begin{equation}
\log P(X)-\mathcal{D}[Q(z) \| P(z \vert X)]=E_{z \sim Q}[\log P(X \vert z)]-\mathcal{D}[Q(z) \| P(z)]
\end{equation}$$

Since we are interested in inferring $$P(X)$$ it would make sense to construct $$Q$$ so that it depends on $$X$$, and a $$Q$$ that would make $$\mathcal{D}[Q(z) \| P(z \vert X)]$$ small 

$$\begin{equation}
\log P(X)-\mathcal{D}[Q(z \vert X) \| P(z \vert X)] = E_{z \sim Q}[\log P(X \vert z)]-\mathcal{D}[Q(z \vert X) \| P(z)]
\end{equation}$$

The left hand is what we want to optimize: $$P(X)$$ and an error term, which is how far our distribution $$Q$$ is from producing $$z$$'s that can reproduce $$X$$ (this distribution is $$\mathcal{D}[Q(z \vert X) \| P(z \vert X)])$$). $$\mathcal{D}[Q(z \mid X) \| P(z \mid X)]$$ is unknown and we cannot evaluate this, which is what makes the left hand side a lower bound. The left hand side is what we optimize via stochastic gradient descent, where $$E_{z \sim Q}[\log P(X \mid z)]$$ is our reconstruction loss and $$\mathcal{D}[Q(z \vert X) \| P(z)]$$ is the KL divergence between our assumed distribution of $$P(z)$$ and the distribution of $$z$$ outputted by our encoder, $$Q(z \vert X)$$.

However, we cannot backpropagate through stochastic layers. To solve this, the "reparameterization trick" is used. The encoder will output a mean vector $$\boldsymbol{\mu}$$ and a covariance matrix $$\boldsymbol{\Sigma}$$. We then sample $$\boldsymbol{\epsilon} \sim \mathcal{N}(0,I)$$ and use this as our $$\boldsymbol{z}$$: $$\boldsymbol{z} = \boldsymbol{\mu} + \boldsymbol{\Sigma}^{1/2} \cdot \boldsymbol{\epsilon}$$. Most often, the covariance matrix is assumed to be diagonal due to computational limits.

In pseudo Python code, this will look like

```python
# log_var is used to not get negative variances
mu, log_var = encoder(X)
# draw random noise from ~N(0,1) for the reparameterization trick
epsilon = randomn(0,1, len(mu)) # could also be len(log_var)
# log_var is assumed to be diagonal here
z = mu + exp(log_var)**(0.5)*epsilon

# reconstruction from the latent space by the decoder
X_hat = decoder(z)

# Reconstruction loss evaluated on a Gaussian distributed (var is a hyperparameter):
rec_loss = Gaussian(X, X_hat, var)

# KL divergence between the distribution from z and a standard Gaussian 
KL_loss = KL(mu, log_var)

ELBO_loss = -(rec_loss - KL_loss)

# backpropagate and step
loss.backward()
optimizer.step()

# Repeat until convergence
```  
&nbsp;

### **Seminal Papers**<a name="#seminal-papers"></a>
* [Tutorial on Variational Autoencoders (2016)](https://arxiv.org/abs/1606.05908)  
A gentle introduction to the theory and intuition of VAEs, with a different proof than using Jensen's inequality for the ELBO loss
* [Semi-Supervised Learning with Deep Generative Models (2014)](https://arxiv.org/abs/1406.5298)  
Introduction of the VAE as a semi-supervised model
* [Auto-Encoding Variational Bayes (2013)](Auto-Encoding Variational Bayes)  
Original paper on VAEs from Kingma and Welling  
&nbsp;

### **Model Variations**<a name="#variations"></a>

* [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework (2017)](https://openreview.net/forum?id=Sy2fzU9gl)  
The authors propose to weight the KL term with hyperparameter $$\beta$$. They show that using $$\beta > 1$$ forces the VAE to learn more disentangled representations.

* [Auxiliary Deep Generative Models (2016)](https://arxiv.org/abs/1602.05473)  
This model adds an extra auxiliary latent to the VAE. As a latent space with a non-diagonal covariance matrix would be heavy to train, the latent is implicitly correlated through the auxiliary latent space, hence making the model more expressive at a lower computational cost.

* [Ladder Variational Autoencoders (2016)](https://arxiv.org/abs/1602.02282)  
This model connects latent layers in a similar fashion to the Ladder Network [Rasmus et al. 2015](https://arxiv.org/abs/1507.02672) and [Valpola 2015](https://arxiv.org/abs/1411.7783). The outputs from the encoder and the sampling for the decoder is shared, which creates a more expressive model with hierarchical latent variables.  
&nbsp;

### **Tricks of the Trade**<a name="#tricks-of-the-trade"></a>
* **Number of Monte Carlo samples from the latent space**: The most common practice is to have single sample from the latent space
* **Size of the latent space**: This needs to be explored empirically. However:
    * [Kingma et al. (2014)](https://arxiv.org/abs/1406.5298) uses 50 for MNIST 
* **KL Warmup**: It is often beneficial to slowly ramp up the KL term (warmup), so it will not dominate the loss
* **Multiple stochastic layers**: [Maaløe et al. (2016)](https://arxiv.org/abs/1602.05473) states that simply cascading two stochastic layers will not converge
&nbsp;


### **Code Repositories**<a name="#code-repositories"></a>
 * **Semi-supervised learning with VAEs** [Github](https://github.com/wohlert/semi-supervised-pytorch)  
 Implementation of several models (VAE, Auxiliary VAE, importance weighted, Ladder VAE, normalizing flows, $$\beta$$-VAE), as well as different sampling strategies.  




