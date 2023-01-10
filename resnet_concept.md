$$
\newcommand{\F}{\mathbb{F}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\F}{\mathcal{F}}
\newcommand{\W}{\hat{W}}
\newcommand{\b}{\hat{b}}
\newcommand{\f}{\hat{f}}
\newcommand{\X}{\mathbf{X}}
\newcommand{\Y}{\mathbf{Y}}
\newcommand{\0}{\mathbf{0}}
\newcommand{\1}{\mathbf{1}}
$$

# ResNet Concept

## Primer

Before **ResNet** paper came out, **VGG** style neural networks were the "cool kids in town". They are usually built by stacking `Conv2d` layers, one after the other. 

Intuitively, the deeper the neural network (i.e. more layers), the "better" the model as it can approximate more functions. However, this is not the case, as pointed out by many empirical evidences as well as the author of the **ResNet** paper.

As we design increasingly deeper networks it becomes imperative to understand how adding layers can increase the complexity and expressiveness of the network.
Even more important is the ability to design networks where adding layers makes networks strictly more expressive rather than just different.

To make some progress we need a bit of mathematics.


## Function Classes

Consider $\mathcal{F}$, the class of functions that a specific network architecture (together with learning rates and other hyperparameter settings) can reach.
That is, for all $f \in \mathcal{F}$ there exists some set of parameters (e.g., weights and biases) that can be obtained through training on a suitable dataset.
Let us assume that $f^*$ is the "truth" function that we really would like to find.
If it is in $\mathcal{F}$, we are in good shape but typically we will not be quite so lucky.
Instead, we will try to find some $f^*_\mathcal{F}$ which is our best bet within $\mathcal{F}$.
For instance, 
given a dataset with features $\mathbf{X}$
and labels $\mathbf{y}$,
we might try finding it by solving the following optimization problem:

$$f^*_\mathcal{F} \stackrel{\mathrm{def}}{=} \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \text{ subject to } f \in \mathcal{F}.$$

It is only reasonable to assume that if we design a different and more powerful architecture $\mathcal{F}'$ we should arrive at a better outcome. In other words, we would expect that $f^*_{\mathcal{F}'}$ is "better" than $f^*_{\mathcal{F}}$. However, if $\mathcal{F} \not\subseteq \mathcal{F}'$ there is no guarantee that this should even happen. In fact, $f^*_{\mathcal{F}'}$ might well be worse. 

As illustrated by :numref:`fig_functionclasses`,
for non-nested function classes, a larger function class does not always move closer to the "truth" function $f^*$. For instance,
on the left of :numref:`fig_functionclasses`,
though $\mathcal{F}_3$ is closer to $f^*$ than $\mathcal{F}_1$, $\mathcal{F}_6$ moves away and there is no guarantee that further increasing the complexity can reduce the distance from $f^*$.
With nested function classes
where $\mathcal{F}_1 \subseteq \ldots \subseteq \mathcal{F}_6$
on the right of :numref:`fig_functionclasses`,
we can avoid the aforementioned issue from the non-nested function classes.


![For non-nested function classes, a larger (indicated by area) function class does not guarantee to get closer to the "truth" function ($f^*$). This does not happen in nested function classes.](http://d2l.ai/_images/functionclasses.svg)
:label:`fig_functionclasses`

Thus,
only if larger function classes contain the smaller ones are we guaranteed that increasing them strictly increases the expressive power of the network.
For deep neural networks,
if we can 
train the newly-added layer into an identity function $f(\mathbf{x}) = \mathbf{x}$, the new model will be as effective as the original model. As the new model may get a better solution to fit the training dataset, the added layer might make it easier to reduce training errors.

This is the question that He et al. considered when working on very deep computer vision models :cite:`He.Zhang.Ren.ea.2016`. 
At the heart of their proposed *residual network* (*ResNet*) is the idea that every additional layer should 
more easily
contain the identity function as one of its elements. 
These considerations are rather profound but they led to a surprisingly simple
solution, a *residual block*.
With it, ResNet won the ImageNet Large Scale Visual Recognition Challenge in 2015. The design had a profound influence on how to
build deep neural networks.

### Interpretation (Important)

The above description was taken from [Dive Into Deep Learning](https://d2l.ai/chapter_convolutional-modern/resnet.html). We try to understand better what they mean here. This will be important to connect the dots later.

First we need to understand Function classes can be all types of functions, polynomial, linear functions etc. In deep learning context, a neural network such as **VGG** is a class of function with many types of hypothesis.

---

Let $\F_1$ be the class of all **linear functions** of the form

$$
\{y=ax+b ~|~ a, b \in \R\}
$$ 

and $\F_2$ be the class of all **quadratic functions** of the form 

$$
\{y = ax^2+bx+c ~|~ a,b,c \in \R \}
$$

We take note that $\F_2$ is "more complex" a model than $\F_1$.

Consider a dataset $\X$ full of images and $\Y$ its labels. We will be less pedantic and say the true function of their underlying relationship is $f(\X) = \Y$. We pretend that $\X$ can be fully predicted by $\Y$ with a linear function $\Y = a\X + b$ where $a = 2, b = 3$. This function $\Y = 2\X + 3$ belongs to $\F_1$.

What the above argument is saying is since $\F_1 \subset \F_2$, then $\F_2$ should at the very least perform as good as $\F_1$. In this case it must be true since we can optimize functions in $\F_2$ such that $a = 0, b = 2, c=3$, which recovers the true function $\Y = 2\X + 3$.

The argument is that $\F_2$ being more complex may not improve model performance or find the "true function" better. It is true if and only if $\F_2$ contains $\F_1$, which is the case here.

If $\F_2$ does not contain $\F_1$, then the image provided above is helpful in visualizing. In the image's context, one sees that the left side of the figure shows a non-nested structure. Imagine $\F_1$ being a small model **AlexNet**, and that is actually close to the ground truth function $f^*$ more than the much more complex model $\F_6$, even though a more complex function/model can contain more hypothesis. In contrast, the right side of the figure shows that if they are nested, then it follows that $\F_6$ should be as near as the ground truth $f^*$ as $\F_1$ is.

> A recap: An analogy to understand this is from polynomial regression. Let’s say I have some data which can be learned effectively using a linear representation, that is, my hypothesis is $h(x)=wx+b$  where $w$ and $b$ are learned parameters. I won't know that a simple linear model would do the trick, so I use a quadratic hypothesis $h(x)=ax^2+bx+c$ while training. Now if the linear hypothesis is the best way to learn this data (i.e. the true function is indeed something like $h(x) = 2x+3$), then one should expect the quadratic hypothesis to learn this linear representation by learning that $a = 0$ as well as $b=2, c=3$.
 
## Vanishing Gradient

[Vanishing Gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) is an earlier problem. 

Extracted from [Detailed Guide to Understand and Implement ResNets](https://cv-tricks.com/keras/understand-implement-resnets/):

> During the backpropagation stage, the error is calculated and gradient values are determined. The gradients are sent back to hidden layers and the weights are updated accordingly. The process of gradient determination and sending it back to the next hidden layer is continued until the input layer is reached. The gradient becomes smaller and smaller as it reaches the bottom of the network. Therefore, the weights of the initial layers will either update very slowly or remains the same. In other words, the initial layers of the network won’t learn effectively. Hence, deep network training will not converge and accuracy will either starts to degrade or saturate at a particular value. Although vanishing gradient problem was addressed using the normalized initialization of weights, deeper network accuracy was still not increasing.


## Degradation Problem

Recall the whole paper was saying that as model complexity rises (more and more layers), the model degrades. However the [Original ResNet paper](https://arxiv.org/abs/1512.03385) did say that model degradation is not due to vanishing gradients, though the previous section has a good argument.


### The problem

[Why does degradation occur in deep neural networks?](https://datascience.stackexchange.com/questions/58568/why-does-degradation-occur-in-deep-neural-networks)

It has been shown that "plain" neural networks tend to have an increased amount *training* error, and accompanied test error, as more layers are added. I am not quite certain as to why this occurs. In the original [ResNet][1] paper they hypothesize and verify that this is **not** due to vanishing gradient. 

From what I understand, it is difficult for a model to approximate the identity map between layers and furthermore when this map is optimal the model may tend to approximate the zero function instead. If this is the case, why does this occur? Finally, why does this not occur in shallower networks in which the identity map may also be optimal?

  [1]: https://arxiv.org/abs/1512.03385

### The answer

The degradation problem has been observed while training deep neural networks. As we increase network depth, accuracy gets saturated (this is expected). Why is this expected? Because we expect a sufficiently deep neural network to model all the intricacies of our data well. There will come a time, we thought, when the extra modelling power provided to us by the additional layers in a deep network will completely learn our data.

This was the easy part. Now we also saw that as we increased the layers of our network further (after the saturation region), the accuracy of the network dropped. Okay, we say, this could be due to overfitting. Except, it’s not due to overfitting, and additional layers in a deep model lead to higher training errors (training, not testing)!

![](https://i.stack.imgur.com/xDLez.jpg)

As you can see in the graph above, deeper networks lead to higher training error. To appreciate how counterintuitive this finding is, consider the following argument.

Consider a network having  n  layers. This network produces some training error. Now consider a deeper network with  m  layers  (m>n) . When we train this network, we expect it to perform at least as well as the shallower network. Why? Replace the first  n  layers of the deep network with the trained  n  layers of the shallower network. Now replace the remaining  n−m  layers in the deeper network with an identity mapping (that is, these layers simply output what is fed into them without changing it in anyway). Thus, our deeper model can easily learn the shallower model’s representation. If there exists a more complex representation of data, we expect the deep model to learn this. See note at the end.

But this doesn’t happen in practice! We just saw above that deeper networks lead to higher training error!

This is the problem residual networks aim to solve.

Note: An analogy to understand this is from polynomial regression. Let’s say I have some data which can be learned effectively using a linear representation, that is, my hypothesis is $h(x)=wx+b$  where $w$ and $b$ are learned parameters. To be extra sure, I use the quadratic hypothesis $h(x)=ax^2+bx+c$  while training. Now if the linear hypothesis is the best way to learn this data, I expect the quadratic hypothesis to learn this linear representation by learning that  $a \to 0$ . This is what is observed in practise, but as we saw above, doesn’t apply to neural networks!

---

## ResNet High-Level Intuition

> Note this is just a high level idea.

### Learning Identity is Hard!

Let us define our problem statement:
- Input: $x=6$;
- True output: $y = 27$;
- The true function is $f(x) = 4x + 3$. For the sake of argument, we must call our true function $f(x) = f_2(f_1(x))$, strictly needing 2 steps (functions) transformation. You will see later why if the true function is a linear function will never really need the so called "skip connection/identity mapping".

We do not truly know the true function so we decided with a neural network of 6 layers. They have naively the following layers and treat each layer as a function (we are being hand-wavy here):

$$
\f_1 \to \f_2 \to \f_3 \to \f_4 \to \f_5 \to \f_6
$$

and also

$$
\f_i = \W_ix + \b_i
$$

where $\W_i$ and $\b_i$ are weights pertaining to the layer $\f_i$. 

Consider our input image is $x$ and output label is $y$, then can we find a transformation such that

$$
y = \f_6(\f_5(\f_4(\f_3(\f_2(\f_1(x)))))) \iff y = 4x + 3
$$

Note this is a composite function and should not come as a surprise since I am trying to mimic a feed-forward neural network here.

Ideally, we only need two steps of transformation to reach our true function, that is if $\f_1$ and $\f_2$ managed to learn the following:

- $\f_1(x) = 2x$ and;
- $\f_2(x) = 2x + 3$;
- It follows that $f_2(f_1(x)) = f_2(2x) = 4x + 3$. More concretely, if our input $x = 6$, then $f_1(x) = 12$ and $f_2(f_1(x)) = f_2(12) = 27$. This is the best scenario and we assume that our network is capable of learning these weights!
- Wait! Oh shoot! We already actually reached our **true output** two layers/functions in! But we cannot just stop here since we have 6 layers! Good god, how'd I wish that all the next 4 layers are just **identity** functions so that $\f_6(\f_5(\f_4(\f_3(...))))$ gives back the $4x+3$, our true function.
- For example, if we put $f_3$ as identity we would have $\f_3(\f_2(\f_1(x))) = \f_3(4x+3) = 4x + 3$ and in turn our output is still 27 since $f_3$ is identity.

> Let us forget about this for a moment and get back to reality. In reality, we have to **learn weights** $\W_i$ and $\b_i$ to form a function $\f_i$ to approximate $f_i$. Ideally, for our first layer/function $\f_1$, we will like to "learn" eventually the weights $\W_1 = 2, \b_1 = 0$ since those are the true weights. We'd also like to learn $\W_2 = 2, \b_2 = 3$ so that $\f_2(x) = 2x + 3$, and subsequently we'd very much like to learn $\W_3 = 1$ and $\b_3 = 0$ so that $\f_3(x) = 1x + 0 = x$ as the identity, we also want to do the same for all the next few layers.

> Now the bummer is that for the model to learn the identity function, it turns out to be very difficult. It seems easy on paper that if you set $\b_3 = 0$ and $\W_3 = 1$ (or identity matrix of 1s in high dimensions), then you can "learn" $\f_3$ to be the identity, but it is not the case at all, learning a matrix of ones or $\W_3 = 1$ in this case is as difficult as learning any other weights...

However, learning zero weights is easy! WHY? Let's see.

### Learning Zero is Easy!

In our usual network (see plain network image in paper), we will learn all 6 layers of weights.

Now, instead of learning $\W_3, \W_4, \W_5, \W_6$ and $\b_3, \b_4, \b_5, \b_6$ directly for $\f_3, \f_4, \f_5, \f_6$, we now define a "skip connection" at say (arbitrary and purposefully) from $\f_3$ to $\f_6$. At $\f_3$, it takes in $\f_2(\f_1(x))$, let us call this output from the first two layers $x_{\text{new}}$.

And define

$$
\f_6(x_{\text{new}}) = F(x_{\text{new}}) + x_{\text{new}}
$$

where in this case $F(x) = \f_5(\f_4(\f_3(x_{\text{new}})))$ and **keep in mind** $x_{\text{new}} = \f_2(\f_1(x))$. What ...... is happenening? How is it even remotely close to learning identity?

The trick is that if $\f_2(f_1(x))$ is truly the optimal weights at $4x+3$, instead of taking my head off learning the identity weights $\W = 1$ and $\b=0$, we force that layer (here is $\f_6$) to learn the previous layers plus its input. In other words, if $F(x_{\text{new}})$ gives you say a big value of $100$ and $x_{\text{new}}$ was $27$, we would have the final output to be $127$, the model will soon realize this pattern and figure that we can push the weights of layers $\f_3$ to $\f_5$ to $0$, then $F(x) = \f_5(\f_4(\f_3(x_{\text{new}})))$ reduces to $0$. This $F(x)$ is the **residual**. Once the model figured out that $F(x)$ can be just $0$, then $\f_6(x_{\text{new}}) = 0 + x_{\text{new}} = 27$, which is what we wanted. In a way, we used a sleight of hand of zero mapping to replace our identity mapping.

To recap, we can either learn slowly and with difficulty, the weights for $\f_3$ to $\f_6$ to be $\W_3 = \W_4 = \W_5 = \W_6 = 1$ and $\b=0$ or use a residual skip connection where we simply define $\f_6(x_{\text{new}}) = F(x_{\text{new}}) + x_{\text{new}}$ and learn the zero mapping.

> If the identity mapping is optimal, we can easily push the residuals to zero (F(x)=0) than to fit an identity mapping (H(x)=x) by a stack of non-linear layers. In simple language it is very easy to come up with a solution like F(x)=0 rather than F(x)=x using stack of non-linear cnn layers as function (Think about it). So, this function F(x) is what the authors called Residual function. - [Extracted Here](https://shuzhanfan.github.io/2018/11/ResNet/)

### Why is Learning Zero Easy?

Most networks initialize weights from a gaussian distribution with mean $0$, this means zero weights are easier to learn! This means if your function is $H(x) = F(x) + x$, where $F(x) = \W x + \b$, then essentially $H(x) = \W x + \b + x$. if $H(x) = x$ is really the optimal answer for the model, then if we initialize our weights and biases to $0$, we will hit $H(x) = 0x + 0 + x = x$ faster and our loss function and optimizer will find out that zero weight is indeed the ideal weight.

### Why is Learning Zero Easy?

Most networks initialize weights from a gaussian distribution with mean $0$, this means zero weights are easier to learn! This means if your function is $H(x) = F(x) + x$, where $F(x) = \W x + \b$, then essentially $H(x) = \W x + \b + x$. if $H(x) = x$ is really the optimal answer for the model, then if we initialize our weights and biases to $0$, we will hit $H(x) = 0x + 0 + x = x$ faster and our loss function and optimizer will find out that zero weight is indeed the ideal weight.

### Why are there two weight layers in one residual block?

https://towardsdatascience.com/intuition-behind-residual-neural-networks-fa5d2996b2c7


## References

### Further Readings

- [Dive Into Deep Learning](https://d2l.ai/chapter_convolutional-modern/resnet.html)
    - We will often cite the text verbatim from this course.
- [Detailed Guide to Understand and Implement ResNets](https://cv-tricks.com/keras/understand-implement-resnets/)
- https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
- https://www.youtube.com/watch?v=GWt6Fu05voI
- https://www.youtube.com/watch?v=RYth6EbBUqM
- https://towardsdatascience.com/intuition-behind-residual-neural-networks-fa5d2996b2c7
- https://shuzhanfan.github.io/2018/11/ResNet/