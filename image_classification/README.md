## Image Classification

The example uses [BVLC-AlexNet](s3.amazonaws.com/store.carml.org/models/tensorflow/models/bvlc_alexnet_1.0/frozen_model.pb) and [synset](http://data.dmlc.ml/mxnet/models/imagenet/synset.txt).

The frozen graph is converted from Caffe BVLC-AlexNet. Refer to [Caffe BVLC-AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) for more details.

### The input and output nodes of the graph

| Node Name    | Input/Output | Shape                     | Data Description                                               |
| ------------ | ------------ | ------------------------- | -------------------------------------------------------------- |
| image_tensor | Input        | [batch, height, width, 3] | RGB pixel values as float32 in a square format (Width, Height) |
| class_scores | Output       | [batch, num_classes]      | Array of probability scores for each class between 0 and 1     |

### Usage

`go run main.go -dir=<model folder> -jpg=<input.jpg> [-labels=<labels.txt>]`

### References
