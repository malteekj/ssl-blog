---
layout: page
title:  "DINO"
date:   #2022-02-01 21:52:57 +0100
permalink:  /contrastive-learning-models/dino/
---

### **Introduction**

**Content**
* [Intuition](#intuition)  
* [Model and Math](#model-and-math)  
* [Implementation and Training Details](#implementation-and-training)  
* [Seminal Papers](#seminal-papers)  
* [Model Variations](#variations)  
* [Tricks of the Trade](#tricks-of-the-trade)  
* [Code Repositories](#code-repositories)  
&nbsp;

### **Intuition**<a name="#intuition"></a>
Self-**di**stillation with **no** labels (DINO) was presented in the paper Emerging Properties in Self-Supervised Vision Transformers from Caron et al. in 2021 at ICCV. It builds upon the idea of self-distillation from BYOL, and utilizes the idea of a teacher and a student network, where the teacher can see global views of an image, but the student only sees local (smaller) views of an image.

![](../../images/contrastive_learning_models/dino.gif){:width="80%"}
*Fig. ?: Schematic of the DINO model. (Image source: [https://github.com/facebookresearch/dino](https://github.com/facebookresearch/dino))*



&nbsp; 

### **Model and Math**<a name="#model-and-math"></a>
A schematic of the DINO architecture can be seen in figure XXX. The architecture consists of a student and a teacher network $$ g_{\theta_s} $$ and $$ g_{\theta_t} $$, which have the exact network architecture, but the teacher network has centering and sharpening step after the embedding. DINO utilizes a multi-crop strategy of a single image, where two global crops $$ x_1^g $$ and $$ x_1^g $$ are constructed along with multiple local crops. All crops are passed through the student while only the global views are passed through the teacher, therefore encouraging “local-to-global” correspondences.


![](../../images/contrastive_learning_models/DINO_arichtecture.png){:width="50%"}
*Fig. ?: Schematic of the DINO model. The student and the teacher are have the exact same architecture, but only the student network has gradients flow through it. (Image source: [Caron et al. 2021](https://arxiv.org/abs/2104.14294))*


An important detail of DINO is that the gradients only flow through the student network, while the parameters of the teacher network is updated by an exponential moving average of the students new weights and the old weights of the teacher:

$$\begin{align}
\theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s
\end{align}$$

With $$ \lambda $$ following a cosine schedule from 0.996 to 1 during training. The authors found that directly copying the student's weights to the teacher does not work, as it leads to convergence issues, while conversely keeping it constant for an entire epoch would work well. \
Originally the momentum encoder was  introduced as a substitute for a queue in contrastive learning, but in DINO it functions as mean teacher in self-training, such as Polyak-Ruppert averaging with an exponential decay. They observe that the teacher had better performance than the student throughout the training, and hence, guides the training of the student by providing target features of higher quality, which was not observed in BYOL, and [Richmond et al](https://arxiv.org/abs/2010.10241). 


The student and teacher network outputs a K dimensional vector, which is converted into a probability distribution $$ P $$ with the softmax function:

$$\begin{align}
P_s(x)^{(i)} = \frac{\exp(g_{\theta_s}(x)^{(i)} / \tau_s)}{\sum_{k=1}^{K} \exp(g_{\theta_s}(x)^{(k)} / \tau_s)}
\end{align}$$

Where $$ \tau_s > 0 $$ is a temperature parameter controlling the sharpness of the distribution. The loss function becomes the cross-entropy between the student and teacher outputs $$ P_s $$ and $$ P_t $$

$$\begin{align}
\min_{\theta_s} H(P_t(x), P_s(x))
\end{align}$$

Where $$ H(a, b) = -a \log b $$. The cross-entropy is calculated between all global and local pairs that are not the same crop:

$$\begin{align}
\min_{\theta_s} \sum_{x \in \{x^g_1, x^g_2\}} \sum_{\substack{x' \in V \\ x' \neq x}} H(P_t(x), P_s(x'))
\end{align}$$

Each batch is centered in teacher by subtracting a mean c. The mean c is updated with an exponential moving average, which allows the approach to work well across different batch sizes:

$$\begin{align}
c \leftarrow mc + (1 - m) \frac{1}{B} \sum_{i=1}^{B} g_{\theta_t}(x_i)
\end{align}$$

In pseudo PyTorch code (without multi-crop) this would look like:
```python
# gs, gt: student and teacher networks 
# C: center (K) 
# tps, tpt: student and teacher temperatures 
# l, m: network and center momentum rates 
gt.params = gs.params 
for x in loader: # load a minibatch x with n samples 
    x1, x2 = augment(x), augment(x) # random views 

    s1, s2 = gs(x1), gs(x2) # student output n-by-K 
    t1, t2 = gt(x1), gt(x2) # teacher output n-by-K 

    loss = H(t1, s2)/2 + H(t2, s1)/2 
    loss.backward() # back-propagate 

    # student, teacher and center updates 
    update(gs) # SGD 
    gt.params = l*gt.params + (1-l)*gs.params 
    C = m*C + (1-m)*cat([t1, t2]).mean(dim=0) 

def H(t, s): 
    t = t.detach() # stop gradient 
    s = softmax(s / tps, dim=1) 
    t = softmax((t - C) / tpt, dim=1) # center + sharpen 
    return - (t * log(s)).sum(dim=1).mean()
```
&nbsp;

### **Implementation and Training Details**<a name="#implementation-and-training"></a>
The authors uses both ResNet50 and ViT to train the model, and reports that ViT works better. They report to surpass supervised ImageNet at 80.1% accuracy with ViT-B and 8x8 pixels patches, which they find to better than 16x16 pixels.

DINO is trained on 16 GPUs and a batch size of 1024 with the AdamW optimizer, where bs 128 gave 57.9% on ImageNet with KNN and a bs of 1024 gave 59.9, so smaller batch sizes can work with more tuning. \
The learning rate was linearly ramped up during the first 10 epochs to its base value determined with the following linear scaling rule: lr = 0.0005 ∗ batchsize/256. After this warmup, the learning rate is decayed with a cosine schedule. 
The weight decay also follows a cosine schedule from 0.04 to 0.4. \
The temperature $$ \tau_s $$ is set to 0.1 while we use a linear warm-up for $$ \tau_t $$ from 0.04 to 0.07 during the first 30 epochs. Notice the temperature is lower for the teacher to have a sharper distribution. They follow the data augmentations of BYOL (color jittering, Gaussian blur and solarization) and multi-crop with a bicubic interpolation to adapt the position embeddings to the scales.

For evaluation, they try both finetuning, linear probing and KNN, since the latter have high reproducibility. They find 20 NN to be most stable.
&nbsp;

### **Seminal Papers**<a name="#seminal-papers"></a>
&nbsp;

### **Model Variations**<a name="#variations"></a>
&nbsp;

### **Tricks of the Trade**<a name="#tricks-of-the-trade"></a>
&nbsp;


### **Code Repositories**<a name="#code-repositories"></a>



