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

![image](https://user-images.githubusercontent.com/38880131/218307932-e91d3b45-ffba-4f30-b291-e953762a42bc.png)

Most information can be found in the uploaded pdf of my master thesis or the repo. However please consider this figure for choosing the LRP-process

![image](https://user-images.githubusercontent.com/38880131/218308328-bec62597-5e75-4fd6-b3f3-2f7781998a48.png)
