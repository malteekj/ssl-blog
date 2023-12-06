---
layout: page
title:  "Contrastive Learning Models"
date:   #2022-02-01 21:52:57 +0100
permalink: /contrastive-learning-models/
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
<!--- Your text here --->
&nbsp; 

### **Model and Math**<a name="#model-and-math"></a>
**Important Models:**
- SimCLR
- BYOL
- CLIP

**Key concepts:**
- Heavy augmentations
    - AutoAugment
    - RandAugment
    - PBA (Population based augmentation)
    - UDA (Unsupervised Data Augmentation)
- Hard negative mining 
    - Debiased Contrastive Learning - Jul 2020 
    - Contrastive Learning with Hard Negative Samples Oct 2021 (ICLR 2021)
- Linear evaluation protocol (see e.g. SimCLR paper): write about how this works and why it is a standard for SSL


- Large batch sizes
- Approaches in Vision
    - Parallel Augmentation (SimCLR, BYOL, Barlow Twins)
    - Memory Bank (MoCo-V2, CURL)
    - Feature Clustering (DeepCluster, SwAV)
    - "Supervised" methods (CLIP)
- Approaches in Language:
    - Text augmentation
    - Back-translation
    - Dropout and cut-off (SimCSE)
    - Supervision from NLI (alot of BERT)
    - Unsupervised Sentence Embedding Learning (context prediction, mutual information)

### **SimCLR**<a name="#seminal-papers"></a>
SimCLR, "A Simple Framework for Contrastive Learning of Visual Representations" Chen et. al. was presented at ICML 2020.

methods:

Consists of the following parts 
 - Stochastic data augmentation: sequentially apply random crop (+ resize back to original size), random color distortion and gaussian blur (crop + color distortion combination is very important)
  - Base encoder f(.) (resnet) which encodes the augmented images into $$ h \in \mathcal{R}^n $$
  - A non-linear projection  head g(h) = z that is a two layer MLP from. The authors found it helpful to apply the contrastive loss on z, and then discard g(.) after training and fine tune on h. z is mostly a 128 dimensional space.
  - the apply contrastive learning loss function (NT-Xent (the normalized temperature-scaled cross entropy loss).) Each example in the batch is augmented into two images. Each augmented pair is treated as a positive pair, and the rest of the pairs in the batch are teated as negative pairs.
  <!-- More math from the paper could be used to describe this section -->

Training:
 - SGD/Momentum was unstable for large batch sizes, so they used the LARS optimizer for all batch sizes and 32-128 TPU cores
 - Global BN was aggregated over all devices (mean and variance) to avoid local exploitation for increased accuracy for the positive pairs, but without getting better representations


Results:
 - 76.5% acc on top-1 with linear evaluation on ImageNet 
 - 85.8% acc on top-5 when finetuned on 1% data on ImageNet 





Key take-aways:
A Simple Framework for Contrastive Learning of Visual Representations
• Composition of multiple data augmentation operations
is crucial in defining the contrastive prediction tasks that
yield effective representations. In addition, unsupervised
contrastive learning benefits from stronger data augmen-
tation than supervised learning.
• Introducing a learnable nonlinear transformation be-
tween the representation and the contrastive loss substan-
tially improves the quality of the learned representations.
• Representation learning with contrastive cross entropy
loss benefits from normalized embeddings and an appro-
priately adjusted temperature parameter.
• Contrastive learning benefits from larger batch sizes and
longer training compared to its supervised counterpart.
Like supervised learning, contrastive learning benefits
from deeper and wider networks.

&nbsp;


&nbsp;

### **Seminal Papers**<a name="#seminal-papers"></a>
&nbsp;

### **Model Variations**<a name="#variations"></a>
&nbsp;

### **Tricks of the Trade**<a name="#tricks-of-the-trade"></a>
&nbsp;

<!-- 
TODO:
 - Understanding the debiasing from lil-log 
 -->


### **Code Repositories**<a name="#code-repositories"></a>



