# TensorFlow Go Examples

This repo contains end-to-end examples of model inference using TensorFlow Go API. More are coming.
Contributions with similar simple examples (or by further improving existing examples) are highly welcome. Please file a pull request or an issue on GitHub.

## Models

- [Image Classification](image_classification): Classify the main object category within an image.

- [Image Object Detection](image_object_detection): Identify the object category and locate the position using a bounding box for every known object within an image.
- [Image Instance Segmentation](image_instance_segmentation): Identify each object instance of each pixel for every known object within an image. Labels are instance-aware.
- [image Semantic Segmentation](image_semantic_segmentation): Identify the object category of each pixel for every known object within an image. Labels are class-aware.
- [image Enhancement](image_semantic_segmentation)

## TensorFlow Go API

Refer to [Install TensorFlow for Go](https://www.tensorflow.org/install/lang_go).

## To inspect pre-trained frozen graphs for input and output tensor names

- [Netron](https://github.com/lutzroeder/netron)
- [How to inspect a pre-trained TensorFlow model](https://medium.com/@daj/how-to-inspect-a-pre-trained-tensorflow-model-5fd2ee79ced0)

## References

- [TensorFlow Go package](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)
- TensorFlow [model repo](https://github.com/tensorflow/models)
