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

![](../images/contrastive_learning/triplet-loss.png)
*Fig. 1: Triplet loss showing positives being pushed closer to the anchor and negative further away. (Image source: [Schroff et al. 2015](https://arxiv.org/abs/1503.03832))*


The triplet loss is given by the following equation:

$$ \begin{equation}
\mathcal{L}_(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-) = \sum_{\mathbf{x} \in \mathcal{X}} \max\big( 0, \|f(\mathbf{x}) - f(\mathbf{x}^+)\|^2_2 - \|f(\mathbf{x}) - f(\mathbf{x}^-)\|^2_2 + \epsilon \big)
\end{equation} $$ 

Where $$ \epsilon $$ is a hyperparameter controlling how much further away $$ \mathbf{x}^- $$ can be from the anchor compared to $$ \mathbf{x}^+ $$. That is, if e.g. $$ \|f(\mathbf{x}) - f(\mathbf{x}^+)\|^2_2 = 1 $$, then $$ \|f(\mathbf{x}) - f(\mathbf{x}^-)\|^2_2 \leq 1 + \epsilon $$. It is crucial to continually mine triplets during training that are in most violation with with the above equation for effective training and convergence.  

<!-- 
The motivation is that the loss
from [14] encourages all faces of one identity to be pro-
jected onto a single point in the embedding space. The
triplet loss, however, tries to enforce a margin between each
pair of faces from one person to all other faces. This al-
lows the faces for one identity to live on a manifold, while
still enforcing the distance and thus discriminability to other
identities. -->





&nbsp;

### **Seminal Papers**<a name="#seminal-papers"></a>
&nbsp;

### **Model Variations**<a name="#variations"></a>
&nbsp;

### **Tricks of the Trade**<a name="#tricks-of-the-trade"></a>
&nbsp;


### **Code Repositories**<a name="#code-repositories"></a>



