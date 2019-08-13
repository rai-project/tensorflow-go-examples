## Image Classification

The example uses [MobileNet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz) and [synset1.txt](http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset1.txt).

Refer to [TensorFlow MobileNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) for more details.

### The input and output nodes of the graph

| Node Name    | Input/Output | Shape                     | Data Description                                               |
| ------------ | ------------ | ------------------------- | -------------------------------------------------------------- |
| image_tensor | Input        | [batch, height, width, 3] | RGB pixel values as float32 in a square format (Width, Height) |
| class_scores | Output       | [batch, num_classes]      | Array of probability scores for each class between 0 and 1     |

### Usage

`go run main.go -dir=<model folder> -jpg=<input.jpg> [-labels=<labels.txt>]`

### References

- https://github.com/Zehaos/MobileNet/blob/master/preprocessing/mobilenet_preprocessing.py