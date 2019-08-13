## Image Super Resolution

The example uses [SRGAN](https://github.com/tensorlayer/srgan/releases/download/1.2.0/g_srgan.npz).

### The input and output nodes of the model

| Node Name    | Input/Output | Shape                                   | Data Description                                              |
| ------------ | ------------ | --------------------------------------- | ------------------------------------------------------------- |
| image_tensor | Input        | [batch, height, width, 3]               | RGB pixel values as uint8 in a square format (Width, Height). |
| output_image | Output       | [batch, output_height, output_width, 3] | Pixel values of the utput enhanced image.                     |

### Usage

Download the frozen model by
```
wget https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/srgan_1.2/frozen_model.pb
```
Or you can export the model youself using the code in `freeze_model` (You need TensorFlow and [TensorLayer](https://github.com/tensorlayer/tensorlayer)). 

Run the inference by

`go run main.go -dir=<model folder> -jpg=<input.jpg> [-out=<output.jpg>] [-labels=<labels.txt>]`

### References

- [Tensorflow Super Resolution Examples](https://github.com/tensorlayer/srgan/) 