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



