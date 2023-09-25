---
layout: page
title:  "Contrastive Learning"
date:   2023-08-04 21:52:57 +0100 #2022-02-01 21:52:57 +0100
permalink: /contrastive-learning/
---

### **Introduction**

**Content**
* [Intuition](#intuition)  
* [Model and Math](#model-and-math)  
* [Seminal Papers](#seminal-papers)  
* [Model Variations](#variations)  
* [Tricks of the Trade](#tricks-of-the-trade)  
* [Code Repositories](#code-repositories)  
&nbsp;

### **Intuition**<a name="#intuition"></a>
The idea behind contrastive learning is to learn an embedding space, where similar inputs will be pushed closer together and dissimilar inputs will be spread further apart. What "similar" refers to, is something the user has to decide on.

The idea of contrastive learning originated as a supervised training method, where the label classes dictated what was similar and dissimilar. While this training method works well, it does not capture the more nuanced relationships between the classes. For instance, the class "cat" and "dog" are semantically different classes, but if "blue whale" is also introduced as a class, it would be reasonable that "cat" and "dog" should be closer to each other (but still apart) in latent space than they would be to "blue whale". Hence, the meaning of similar and dissimilar is often not discrete, but rather a spectrum.

Continuing on this thought, one would rightly ask: How do we then define how much more similar "cat" and "dog" should be than "blue whale"? This question is not easy to answer. Unless we have a specific reason to hardcode this relative relationship, a good solution would be to simply let the data speak for it self, without us imposing specific classes from the beginning. This leads to the self-supervised formulation of contrastive learning.

In this case, we construct tasks where our network will learn features and correlations in our data, but in a more generalized manner. So similar will mean how much the network can compress the input and still solve the given task, and will hence be data driven rather than introduced as a bias from us.

Contrastive loss was first defined by [Bromley et al](https://proceedings.neurips.cc/paper/1993/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf) **(a bit more history)**.

&nbsp; 

### **Model and Math**<a name="#model-and-math"></a>
As described in above, contrastive learning can both be supervised or self-supervised. The main difference is how the negative and positive pairs are defined, where in supervised learning they are labelled by hand and in self-supervised learning they are mined.

The objective is to learn a function $$ f_\theta(.): \mathcal{X}\to\mathbb{R}^d $$ for a given set of data points $$ \{ \mathbf{x}_i \} $$, where $$ f_\theta(.) $$ encodes $$x_i$$ into an embedding space, where similar classes are pushed closer together and dissimilar classes further apart. Similar will either refer to a set of corresponding labels $$y_i \in \{1, \dots, L\}$$ for $$ \{ \mathbf{x}_i \} $$ in a supervised setting, otherwise it will depend on a mining heuristic.


#### **_Paired Contrastive Loss_** 
Introduced in [Chopra et al 2005](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf), which is one of the earliest uses of the contrastive loss. Here, the contrastive loss takes in pairs of input data points $$ (x_i, x_j) $$ and minimizes the distance between positive pairs in embedding space while maximizing the distance of negative pairs.

$$ \begin{equation} \mathcal{L}(\mathbf{x}_i, \mathbf{x}_j, \theta) = \mathbb{1}[y_i=y_j] \| f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j) \|^2_2 + \mathbb{1}[y_i\neq y_j]\max(0, \epsilon - \|f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j)\|_2)^2 \end{equation} $$

Where $$ \epsilon $$ is a hyperparameter that controls the maximum distance between negative pairs, which essentially becomes as target distance for negative pairs.

#### **_Triplet Loss_** 
Triplet loss uses three data points as input: $$ (\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-) $$, where $$ \mathbf{x} $$ is called the anchor point, and $$ \mathbf{x}^+ $$ and $$ \mathbf{x}^- $$ are a positive and negative pair with $$ \mathbf{x} $$, respectively. It was introduced in FaceNet by [Schroff et al. 2015](https://arxiv.org/abs/1503.03832), where it was used to train a face recognition model of the same person in different poses and angles. The motivation for the triplet loss is to avoid that the classes will be projected onto a single point (as paired contrastive loss encourages), and rather enforce a margin between positive pairs within a class and to the other classes.

![](../images/contrastive_learning/triplet-loss.png){:width="70%"}
*Fig. 1: Triplet loss showing positives being pushed closer to the anchor and negative further away. (Image source: [Schroff et al. 2015](https://arxiv.org/abs/1503.03832))*


The triplet loss is given by the following equation:

$$ \begin{equation}
\mathcal{L}_(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-) = \sum_{\mathbf{x} \in \mathcal{X}} \max\big( 0, \|f(\mathbf{x}) - f(\mathbf{x}^+)\|^2_2 - \|f(\mathbf{x}) - f(\mathbf{x}^-)\|^2_2 + \epsilon \big)
\end{equation} $$ 

Where $$ \epsilon $$ is a hyperparameter controlling how much further away $$ \mathbf{x}^- $$ can be from the anchor compared to $$ \mathbf{x}^+ $$. That is, if e.g. $$ \|f(\mathbf{x}) - f(\mathbf{x}^+)\|^2_2 = 1 $$, then $$ \|f(\mathbf{x}) - f(\mathbf{x}^-)\|^2_2 \leq 1 + \epsilon $$. It is crucial to continually mine triplets during training that are in most violation with with the above equation for effective training and convergence.  

#### **_Lifted Structural Loss_** 
Lifted structural loss compares the distance between a positive pair $$ (\mathbf{x}_i, \mathbf{x}_j) \in \mathcal{P} $$ with all observations that forms negative pairs with them within a training batch $$ \mathbf{x}_k \in \mathcal{N} $$ where, respectively, $$ \mathcal{P} $$ is the set of positive pairs and $$ \mathcal{N} $$ is the set of negative pairs for at given positive pair in $$ \mathcal{P} $$. It was introduced by [Song et al. 2015](https://arxiv.org/abs/1511.06452).

![](../images/contrastive_learning/lifted-structured-loss.png){:width="45%"}
*Fig. 2: Illustrates the difference between paired contrastive loss, triplet loss and lifted structured loss. (Image source: [Song et al. 2015](https://arxiv.org/abs/1511.06452))*

The lifted structured loss is formulated as:

$$ \begin{aligned}
\mathcal{L} &= \frac{1}{2\vert \mathcal{P} \vert} \sum_{(i,j) \in \mathcal{P}} \max(0, \mathcal{L}_\text{struct}^{(ij)})^2 \\
\text{where } \mathcal{L}^{(ij)} &= D_{ij} + {\max \big( \max_{(i,k)\in \mathcal{N}} \epsilon - D_{ik}, \max_{(j,l)\in \mathcal{N}} \epsilon - D_{jl} \big)} \\
\text{and } D_{ij} &= \vert f(\mathbf{x}_i) - f(\mathbf{x}_j) \vert_2
\end{aligned} $$

All the pairwise distances are calculated within each training batch. Notice that $$ {\max \big( \max_{(i,k)\in \mathcal{N}} \epsilon - D_{ik}, \max_{(j,l)\in \mathcal{N}} \epsilon - D_{jl} \big)} $$ is used mine the observation that has the greatest distance to the positive pair $$ (\mathbf{x}_i, \mathbf{x}_j) $$, which just like in triplet loss is important for effective training and convergence. Further, it is not a smooth function which can lead to problems which convergence and ending in a poor local optimum. The authors therefore relax this part to:

$$ 
\mathcal{L}^{(ij)} = D_{ij} + \log \Big( \sum_{(i,k)\in\mathcal{N}} \exp(\epsilon - D_{ik}) + \sum_{(j,l)\in\mathcal{N}} \exp(\epsilon - D_{jl}) \Big)
$$

The authors emphasize the importance of incorporating hard examples into each batch for better convergence. \<a little more about the details in the paper\>

#### **_N-pair Loss_** 
Multi-class N-pair loss extends the idea of triplet loss to compare the anchor point to multiple negative samples rather than one negative pair. It was introduced by [Sohn 2016](https://papers.nips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html).

![](../images/contrastive_learning/N_pair_loss.png){:width="85%"}
*Fig. 2: Illustration of the difference between triplet loss (left) and N-pair loss (right). Red dots are negative pairs and blue dots are positive pairs. Notice this figure is different form the original one which had some typos. (Image source: [Sohn 2016](https://papers.nips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html))*

In the N-pair loss we have a training batch consisting of a $$ (N + 1) $$ tuplet $$ \{ \mathbf{x}, \mathbf{x}^+, \mathbf{x}^-_1, \dots, \mathbf{x}^-_{N-1} \} $$ with one anchor point, one sample that constitutes a positive pair with the anchor and $$ N-1 $$ negatives samples that constitutes negative pairs. The loss is defined as:

$$ \begin{aligned}
\mathcal{L}(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-_1, \dots, \mathbf{x}^-_{N-1}) 
&= \log\big(1 + \sum_{i=1}^{N-1} \exp(f(\mathbf{x})^\top f(\mathbf{x}^-_i) - f(\mathbf{x})^\top f(\mathbf{x}^+))\big) \\
&= -\log\frac{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+))}{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+)) + \sum_{i=1}^{N-1} \exp(f(\mathbf{x})^\top f(\mathbf{x}^-_i))}
\end{aligned} $$

Where $$ f(\mathbf{x}) $$ is the embedding function of the observation (usually a neural network). The motivation behind the N-pair loss is that when using the triplet loss, we are only guaranteed to be far from the selected negative sample but not necessarily the other negative samples. Hence, we risk only differentiating the positive and negative pairs from few samples, while other negative samples might still remain closer to the positive one. This helps stabilize and speed up convergence.

The loss collapses into normal softmax loss for a multi-class classification if only one negative sample is made per class. Notice the author uses the (un-normalized) inner product of the embedding vectors as the distance measure, but regularize the $$ L^2 $$ norm of the embedding vectors instead of normalizing.  

#### **_Noise Contrastive Estimation (NCE)_** 
NCE is a contrastive method where the distance between pairs is formulated as a logistic regression problem rather than a metric distance. We are contrasting a target class $$ \mathbf{x} $$ with a noise class $$ \mathbf{\tilde x} $$, which in a supervised setting corresponds to a binary classification problem. In SSL, this would correspond to negative and positive pairs we would have to mine. It was introduced by [Gutmann & Hyvarinen 2010](http://proceedings.mlr.press/v9/gutmann10a.html)

In this setting, the embedding function, called $$  p_m^0(\;\cdot \; ;\alpha) $$ in the paper, does not necessary integrate to 1 (i.e. is not a probability distribution), akin to an energy function. The normalization constant that makes it sum to 1 is thus part of the parameters to be optimized, i.e. $$  p_m(\;\cdot \; ;\theta) = p_m^0(\;\cdot \; ;\alpha)/c $$, where $$ \theta = \{\alpha, c\} $$. The point here is that we do not need to chose a probability function upfront, but can rather learn an embedding function and normalize as part of the optimization. This makes the model more flexible.

We assume that the target class follows the distribution $$ \mathbf{x} \sim P(\mathbf{x} \vert C=1; \theta) = p_m(\mathbf{x};\theta) $$ and the noise distribution $$ \mathbf{\tilde x} \sim  P(\tilde{\mathbf{x}} \vert C=0) = p_n(\tilde{\mathbf{x}}) $$

We are modelling the logits of a sample $$ \mathbf{u} $$ belonging to the target distribution:

$$ \begin{aligned}
G(\mathbf{u}) = \log \frac{p_m(\mathbf{x};\theta)}{p_n(\mathbf{x}) } = \log p_m(\mathbf{x};\theta) - \log p_n(\mathbf{x}) 
\end{aligned} $$

We use the logistic function to uptain probabilities for each classes

$$ \begin{aligned}
\sigma(G) &= \frac{1}{1 + \exp(-G)} =  \frac{p_m(\mathbf{x};\theta)}{p_m(\mathbf{x};\theta) + p_n(\mathbf{x}) }
\end{aligned} $$

The objective function then becomes:

$$ \begin{aligned}
\mathcal{L} = - \frac{1}{2N} \sum_{i=1}^N \big[ \log \sigma (G(\mathbf{x}_i)) + \log (1 - \sigma (G(\tilde{\mathbf{x}}_i))) \big] 
\end{aligned}
$$

The objective function is hence maximizing the discrimination between the target distribution and noise distribution. In SSL, this would correspond to discriminating between the positive and negative samples. 

<!-- - We are modelling the samples and noise by a logistic regression, to tell then apart
- This is modelled as their log odds ratio
    - but why not just model it as a probability? -->


#### **_InfoNCE_** 
InfoNCE aims at maximizing the mutual information between the embedding of the next patches in a signal and the prediction of the embedding from the current patch. By predicting the latent representation of the next patch, the patch does not need to be fully reconstructed for training. It was introduced by [van den Oord et al. 2018](https://arxiv.org/abs/1807.03748). The reasoning behind this is that both signals and images often contain high frequency features (i.e. very local details), which are not important for a classification task. In generative models that uses e.g. MSE as loss, these high frequency details are also reconstructed, which is wasteful and possibly detrimental for downstream classification. 

The model work by taking in a signal or image $$ \mathbf{X} $$ and dividing it into patches $$ {\mathbf{x}_t} $$. Each patch is then encoded into a latent representation $$ g_{enc}(\mathbf{x}_t) = \mathbf{z}_t $$, and an autoregressive model takes in $$ \mathbf{z}_{\leq t} $$ and outputs a context vector $$ \mathbf{c}_t $$, $$ g_{ar}(\mathbf{z}_{\leq t}) = \mathbf{c}_t $$. A positive sample is sampled from the conditional distribution $$ p(\mathbf{x}\vert \mathbf{c}) $$ and $$ N-1 $$ negative samples are sampled from the noise distribution $$ p(\mathbf{x}) $$ in each batch.

Using Bayes' Theorem, the probability of detecting the positive sample is:

$$\begin{align}
p(C=\texttt{pos} \vert X, \mathbf{c}) 
= \frac{p(x_\texttt{pos} \vert \mathbf{c}) \prod_{i=1,\dots,N; i \neq \texttt{pos}} p(\mathbf{x}_i)}{\sum_{j=1}^N \big[ p(\mathbf{x}_j \vert \mathbf{c}) \prod_{i=1,\dots,N; i \neq j} p(\mathbf{x}_i) \big]}
= \frac{ \frac{p(\mathbf{x}_\texttt{pos}\vert c)}{p(\mathbf{x}_\texttt{pos})} }{ \sum_{j=1}^N \frac{p(\mathbf{x}_j\vert \mathbf{c})}{p(\mathbf{x}_j)} }
= \frac{f(\mathbf{x}_\texttt{pos}, \mathbf{c})}{ \sum_{j=1}^N f(\mathbf{x}_j, \mathbf{c}) }
\end{align}$$

Hence the scoring function is proportional to: $$ f(\mathbf{x}, \mathbf{c}) \propto \frac{p(\mathbf{x}\vert\mathbf{c})}{p(\mathbf{x})} $$. The InfoNCE loss optimizes the negative log probability of classifying the positive sample correctly:

$$\begin{align}
\mathcal{L} = - \mathbb{E} \Big[\log \frac{f(\mathbf{x}, \mathbf{c})}{\sum_{\mathbf{x}' \in X} f(\mathbf{x}', \mathbf{c})} \Big]
\end{align}$$

Since $$ f(\mathbf{x}, \mathbf{c}) $$ estimates the density ratio $$ \frac{p(x\vert c)}{p(x)} $$ it can be related to mutual information optimization. The mutual information between input $$ \mathbf{x} $$ and context vector $$ \mathbf{c} $$ can be rewritten, such that the relation to $$ \frac{p(\mathbf{x}\vert \mathbf{c})}{p(\mathbf{x})} $$ becomes clear:

$$\begin{align}
I(\mathbf{x}; \mathbf{c}) = \sum_{\mathbf{x}, \mathbf{c}} p(\mathbf{x}, \mathbf{c}) \log\frac{p(\mathbf{x}, \mathbf{c})}{p(\mathbf{x})p(\mathbf{c})} = \sum_{\mathbf{x}, \mathbf{c}} p(\mathbf{x}, \mathbf{c})\log{\frac{p(\mathbf{x}|\mathbf{c})}{p(\mathbf{x})}}
\end{align}$$

As the described earlier, we want to avoid modeling the future observations $$ p_k(\mathbf{x}_{t+k} \vert \mathbf{c}_t) $$ directly (i.e. generate/reconstruct the original signal). Rather, we CPC models a density function to preserve the mutual information between $$ \mathbf{x}_{t+k} $$ and $$ \mathbf{c}_t $$:

$$\begin{align}
f_k(\mathbf{x}_{t+k}, \mathbf{c}_t) \propto \frac{p(\mathbf{x}_{t+k}\vert\mathbf{c}_t)}{p(\mathbf{x}_{t+k})}
\end{align}$$

Notice here that the ratio between the two densities $$ f $$ does not have to be normalized (i.e. integrate to 1). Any positive real score can be used. The authors chose to use a log-bilinear model:

$$\begin{align}
f_k(\mathbf{x}_{t+k}, \mathbf{c}_t) = \exp(\mathbf{z}_{t+k}^\top \mathbf{W}_k \mathbf{c}_t)
\end{align}$$

Where $$ \mathbf{W}_k \mathbf{c}_t $$ becomes the prediction of $$ \mathbf{z}_{t+k} $$, i.e. $$ \mathbf{\tilde{z}}_{t+k} = \mathbf{W}_k \mathbf{c}_t $$ (with a different $$ \mathbf{W}_k $$ for each timestep k), and the inner product between $$ \mathbf{z}_{t+k} $$ and $$ \mathbf{\tilde{z}}_{t+k} $$ thus becomes the similarity between the prediction of the latent representation at time step $$ t+k $$ and the embedded latent representation.

<!-- 
Questions:
 - How is W_k used and generated? Is it recursively updated or outputted by the network?
 - Since we are essentially modelling the inner product of the z and z_pred, what exactly makes
    this a probabilistic model?

 -->


&nbsp;

### **Seminal Papers**<a name="#seminal-papers"></a>

#### _Constrastive loss_
- [Learning a Similarity Metric Discriminatively, with Application to Face
Verification (Chopra et al. 2005)](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)
    - Simple paired contrastive loss.
- [FaceNet: A Unified Embedding for Face Recognition and Clustering (Schroff et al. 2015)](https://arxiv.org/abs/1503.03832)
    - Triplet loss with an anker point and a positive and negative sample.
- [Deep Metric Learning via Lifted Structured Feature Embedding (Song et al. 2015)](https://arxiv.org/abs/1511.06452)
    - Lifted structural loss with comparison among all inputs in the batch.
- [Improved Deep Metric Learning with Multi-class N-pair Loss Objective (Sohn 2016)](https://papers.nips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html)
    - N-pair loss.
- [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models (Gutmann & Hyvarinen 2010)](http://proceedings.mlr.press/v9/gutmann10a.html)
    - Noise contrastive estimation (NCE).
- [Representation Learning with Contrastive Predictive Coding (van den Oord et al. 2018)](https://arxiv.org/abs/1807.03748)
    - InfoNCE (builds upon NCE with mutual information)



&nbsp;

### **Model Variations**<a name="#variations"></a>
&nbsp;

### **Tricks of the Trade**<a name="#tricks-of-the-trade"></a>
- \<choosing sampling strategies for choosing hard examples within a batch for better convergence\>


<!-- questions:
- why is the focus on using one positive example against several negatives rather than multiple of each?
 -->


&nbsp;


### **Code Repositories**<a name="#code-repositories"></a>



