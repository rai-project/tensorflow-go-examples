## Image Semantic Segmentation

The example uses DeepLab [mobilenetv2_coco_voc_trainaug](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz) trained on PASCAL VOC 2012.

Refer to [TensorFlow DeepLab Model Zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) and [DeepLab: Deep Labelling for Semantic Image Segmentation](https://github.com/tensorflow/models/tree/master/research/deeplab) for more details.

### The input and output nodes of the model

| Node Name            | Input/Output | Shape                     | Data Description                                             |
| -------------------- | ------------ | ------------------------- | ------------------------------------------------------------ |
| ImageTensor          | Input        | [batch, height, width, 3] | RGB pixel values as uint8 in a square format (Width, Height) |
| SemanticPredictionss | Output       | [batch, outeight, width]  | Array of output segments                                     |

### Usage

`go run main.go -dir=<model folder> -jpg=<input.jpg> [-out=<output.jpg>] [-labels=<labels.txt>]`

### References

- [DeepLab Demo](https://github.com/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
- [](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)
- TensorFlow [visualization_utils](https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py)
- [An overview of semantic image segmentation](https://www.jeremyjordan.me/semantic-segmentation/)