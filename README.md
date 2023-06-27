# lrp-based-attacks

In this repository, I document the results of my Master Thesis about adversarial attacks that are based on explainable AI methods, more precisely here: Layerwise Relevance Propagation (https://www.hhi.fraunhofer.de/en/departments/ai/technologies-and-solutions/layer-wise-relevance-propagation.html).

## More about this implementation

In this repository, I implemented two main functions:

- Layerwise Relevance Propagation (LRP) for any tensorflow based model
- Adversarial attacks based on the LRP implementation here, in three ways:
    - flip: Pixels are flipped (multiplied by -1, due to value range [-1,...,1])
    - mean: Pixels are shifted towards pixel mean value of the image
    - lrp-grad: Pixels are manipulated along their gradient, however chosen according to relevance
- The attacks have been implemented as single or batch pixel manipulations
- Early stopping can be activated to stop manipulation as soon as the predicted label has changed


The implementation is based on tensorflow for now, however I plan to extent it to pytorch
Libraries used:
- python      3.10.8
- numpy       1.21.5
- pandas      1.5.1
- tensorflow   2.10.0

![](https://github.com/robisc/lrp-based-attacks/blob/master/images/CIFAR10_LRPgrad_example_img.gif)
