# Helpsheet: Convolutional Neural Networks

Everything here is relevant to 2D convolutional neural networks. Definitions may need to scale up
or down if we are talking about 1D or 3D convolutional neural networks.

```{prf:definition} Image
:label: def_image

We define image $\X$ of size $C \times H \times W$ where $C$ is the number of channels,
$H$ is the height and $W$ is the width of the image.
```

## Properties of CNNs

To design a Convolutional Neural Networks, we should bear these properties in mind.

```{prf:property} Translational Invariance/Translational Equivariance
:label: def_translational_invariance

In the earliest layers, our network should respond similarly to the same patch, regardless of where it appears in the image. This principle is called translation invariance (or translation equivariance) {cite}`zhang2021dive`.

This is why we often say the earlier layers of a CNN detects shapes and edges.
```

```{prf:property} Locality
:label: def_locality

The earliest layers of the network should focus on local regions, without regard for the contents of the image in distant regions. This is the locality principle. Eventually, these local representations can be aggregated to make predictions at the whole image level {cite}`zhang2021dive`.
```

```{prf:property} Abstractness
:label: def_abstractness

As we move deeper into the network, the representations should become more abstract and less sensitive to the exact location of objects in the image. This is the abstractness principle.
```

```{prf:property} Parameter/Weight Sharing
:label: def_parameter_sharing

CNNs share parameters across space. This allows the network to learn features that are useful in multiple locations. This is the parameter sharing principle.

One can imagine that the first few layers are detecting shapes and edges, and therefore this same edge detector 
will still be useful as move deeper into the network. Therefore, we can say we are sharing parameters across space.
```

```{prf:remark} Further Reading
:label: def_cnn_properties_remark

Note that we are being loose with terms such as Translational Invariance and Translational Equivariance.
One can read more about them [here](https://datascience.stackexchange.com/questions/16060/what-is-the-difference-between-equivariant-to-translation-and-invariant-to-tr#:~:text=Equivariant%20to%20translation%20means%20that,an%20equivalent%20translation%20of%20outputs.) and [here](https://towardsdatascience.com/translational-invariance-vs-translational-equivariance-f9fbc8fca63a).
```

## Cross Correlation

```{prf:definition} Cross Correlation
:label: def_cross_correlation

$\X$
```

````{figure} https://storage.googleapis.com/reighns/reighns_ml_projects/docs/gaohn-machine-learning-foundations/correlation_d2l.svg
---
name: fig_cross_correlation
---
Two-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: 
$0 \times 0 + 1 \times 1 + 3 \times 2 + 4 \times 3 = 19$. 

Image Credits: [Dive into Deep Learning](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html)
````

## Filters and Kernels

```{prf:definition} Kernel
:label: def_kernel

A **kernel** is a 2D matrix that is used to extract features from an image $\X$. 

Although no strict restrictions on the kernel size, the kernel is typically a square
of size $K \times K$.
```

```{prf:definition} Filter
:label: def_filter

A **filter** is a 3D matrix that is a collection of kernels stacked together.

A filter is typically of size $C_{\text{in}} \times K \times K$ where $C_{\text{in}}$ is the number of channels
in the preceding layer.
```

```{prf:definition} Output Size
:label: def_output_size

Given an input size of $n_h \times n_w$, and its corresponding kernel size to be $k_h \times k_w$, 
the output size of a convolutional layer is given by the following formula:

$$
(n_h - k_h + 1) \times (n_w - k_w + 1)
$$
```



## Feature Map and Receptive Field

```{prf:definition} Feature Map
:label: def_feature_map

The output of a convolutional layer is often called a **feature map** {cite}`zhang2021dive`.

The example in {numref}`fig_cross_correlation` shows the output to be our feature map.
```

```{prf:definition} Receptive Field
:label: def_receptive_field

In the context of Convolutional Neural Networks, the **receptive field** is the region of the input image that a single neuron in the convolutional layer is connected to.

More concretely, for any element $x$ in a layer, the receptive field is the region of the input image that contributed to the computation of $x$ {cite}`zhang2021dive`.

In {numref}`fig_cross_correlation`, the receptive field of the first element in the output is the shaded region in the input.
```


## Padding

Notice that the output size defined in {prf:ref}`def_output_size` is smaller than the input size. Therefore,
after numerous convolutional layers, the output size will be very small. Furthermore, the edges
of the image are used much less than the center of the image. We can use padding to fix these issues.

```{prf:definition} Valid Padding
:label: def_valid_padding

Valid padding is when we do not pad the input image. 
```

```{prf:definition} Same Padding
:label: def_same_padding

Same padding is when we pad the input image such that the output size is the same as the input size.
We usually pad with zeros.

Given an image $\X$ of size $n_h \times n_w$, if we pad $p$ additional cells on each side of the image, then our input image $\X$ will now have
an input size of $(n_h + 2p) \times (n_w + 2p)$. If we apply a convolutional layer with a kernel size of $k_h \times k_w$,
we end up with an output size of $(n_h + 2p - k_h + 1) \times (n_w + 2p - k_w + 1) = n_h \times n_w$ by {prf:ref}`def_output_size`.

Then, recall our purpose is to pad the input image such that the output size is the same as the input size. Therefore, we can solve for $p$:

$$
n_h + 2p - k_h + 1 = n_h \implies p = \frac{k_h - 1}{2} = \frac{k - 1}{2}
$$

We have assumed that the image and kernel are square in size. If they are not, then we can solve for $p$ separately for each dimension.
Further, we assumed that $k_h = k_w$ is odd, if not, we may to apply floor and ceiling to the above formula.
```

## Stride

```{prf:definition} Stride
:label: def_stride

In the context of Convolutional Neural Networks, the **stride** is the number of pixels we shift the kernel over the image.
```

## Feature Map (Output) Dimensions

```{prf:definition} Calculating Output Dimensions
:label: def_calculating_output_dimensions

Given an input image $\X$ of size $n \times n$, a kernel of size $k \times k$, 
a padding of size $p$, and a stride of size $s \times s$, the output dimensions are given by:

$$
\lpar \lfloor \frac{n + 2p - k}{s} \rfloor + 1 \rpar \times \lpar \lfloor \frac{n + 2p - k}{s} \rfloor + 1 \rpar
$$

where $\lfloor x \rfloor$ is the floor function.
```

## Convolutions over Volumes

So far we have only discussed convolutions over 2D images. However, we can also apply convolutions over 3D volumes (i.e. RGB images with 3 channels).


For example, we can apply convolutions over a 3D volume of size $n_h \times n_w \times n_c$ where $n_c$ is the number of channels. In this case, the kernel is also a 3D volume of size $k_h \times k_w \times n_c$.
<link rel="stylesheet" type="text/css" href="https://tikzjax.com/v1/fonts.css">
<script src="https://tikzjax.com/v1/tikzjax.js"></script>


<script type="text/tikz">
  \begin{tikzpicture}[scale=0.5]     \draw (0.5, 0.5) rectangle (3.5, 3.5);     \draw (0.25, 0.25) rectangle (3.25, 3.25); 
    \draw (0, 0) rectangle (3, 3);
    \node at (1.5, -.5) {$6 \times 6$};
    \node (channels) at (6.75, .4) {$3$ channels};
    \draw[->] (channels) to (3.5, .4);
  \end{tikzpicture} 
</script>

## Summary

```{prf:remark} Learning Filters
:label: def_learning_filters_remark

Hand picking kernels and filters can be a tedious task. Instead, we let the CNN learn
the filters and kernels for us. Consequently, all the elements in the filter are learnable parameters.
```

```{prf:remark} Filter Size
:label: def_filter_size_remark

A common misconception arises when we say that we are using a filter of size, say $5 \times 5$.
This **does not mean** the filter is a square of size $5 \times 5$, instead, it means the filter is a 3D matrix of size $C_{\text{in}} \times 5 \times 5$.
```








````{figure} https://storage.googleapis.com/reighns/reighns_ml_projects/docs/gaohn-machine-learning-foundations/cs231n-convolutional-demo.gif
---
name: fig_filters_and_kernels
---
The filter is applied to each region of the input image, and the result is a single value. Image Credit: [CS231N](https://cs231n.github.io/convolutional-networks/)
````



## Further Readings

- Zhang, Aston, Zachary C. Lipton, Mu Li, and Alexander J. Smola. "Chapter 7. Convolutional Neural Networks." In Dive into Deep Learning. Berkeley: 2021

