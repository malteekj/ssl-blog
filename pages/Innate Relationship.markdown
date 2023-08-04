---
layout: page
title:  "Innate Relationship"
date:   2022-10-08 +0100 #2022-02-01 21:52:57 +0100
permalink: /innate_relationship/
---

### **Introduction**
Innate relationship is the process of pretraining a model by solving a hand-crafted task which can leverage the internal structure of the data. This task is different from other strategies, such as optimizing the model’s reconstruction (generative and self-prediction) or the latent representation of the original input (contrastive learning and [variational autoencoders](/ssl-blog/VAEs)). Examples of innate relationship for e.g. visual inputs include predicting image rotation angle, flip prediction, solving jigsaw puzzles of an image, or determining the relative positions of image patches.

The model learn features on from data by solving the hand-crafted task, which serves as a starting point for finetuning on a downstream task. However, how well the learned features will generalize will depend strongly on the specifics of the hand-crafted task. If the chosen task is not well suited for the downstream problem, the learned features might only be effective for the task but have limited benefits for the downstream task.

Innate relationship is one of earlier methods for SSL and is also one of the more simple SSL approaches. More powerful SSL methods has later been developed, but innate relationship still have some relevance for SSL.

**Content**
* [Intuition](#intuition)  
* [Model and Math](#model-and-math)  
* [Seminal Papers](#seminal-papers)  
* [Model Variations](#variations)  
* [Tricks of the Trade](#tricks-of-the-trade)  
* [Code Repositories](#code-repositories)  
&nbsp;

### **Intuition**<a name="#intuition"></a>
&nbsp; 

### **Model and Math**<a name="#model-and-math"></a>
&nbsp;

### **Seminal Papers**<a name="#seminal-papers"></a>
&nbsp;

### **Model Variations**<a name="#variations"></a>
&nbsp;

### **Tricks of the Trade**<a name="#tricks-of-the-trade"></a>
&nbsp;


### **Code Repositories**<a name="#code-repositories"></a>



