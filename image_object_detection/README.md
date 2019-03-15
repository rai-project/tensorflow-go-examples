## Object Detection

The reference output image is generated with [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz), a multi-object detection model trained on the COCO dataset.

Refer to [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/g3doc/detection_model_zoo.md) for more details.

### The input and output nodes of the graph

| Node Name         | Input/Output | Shape     | Data Description                                                                                         |
| ----------------- | ------------ | --------- | -------------------------------------------------------------------------------------------------------- |
| image_tensor      | Input        | [1,?,?,3] | RGB pixel values as uint8 in a square format (Width, Height). The first column represent the batch size. |
| detection_boxes   | Output       | [?][4]    | Array of boxes for each detected object in the format [yMin, xMin, yMax, xMax]                           |
| detection_scores  | Output       | [?]       | Array of probability scores for each detected object between 0..1                                        |
| detection_classes | Output       | [?]       | Array of object class indices for each object detected based on COCO objects                             |
| num_detections    | Output       | [1]       | Number of detections                                                                                     |

### Usage

`go run main.go -dir=<model folder> -jpg=<input.jpg> [-out=<output.jpg>] [-labels=<labels.txt>]`

### Reference
- [gococo](https://github.com/ActiveState/gococo)
