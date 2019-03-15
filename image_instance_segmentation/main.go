package main

import (
	"flag"
	"image"
	"image/draw"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"golang.org/x/image/colornames"
)

func main() {
	// Parse flags
	modeldir := flag.String("dir", "", "Directory containing trained model files. Assumes model file is called frozen_inference_graph.pb")
	jpgfile := flag.String("jpg", "lane_control.jpg", "Path of a JPG image to use for input")
	outjpg := flag.String("out", "output.jpg", "Path of output JPG for displaying labels. Default is output.jpg")
	labelfile := flag.String("labels", "coco_labels.txt", "Path to file of COCO labels, one per line")
	flag.Parse()
	if *modeldir == "" || *jpgfile == "" {
		flag.Usage()
		return
	}

	// Load the labels
	labels := utils.LoadLabels(*labelfile)

	// Load a frozen graph to use for queries
	modelpath := filepath.Join(*modeldir, "frozen_inference_graph.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, i, err := utils.MakeTensorFromImage(*jpgfile)
	if err != nil {
		log.Fatal(err)
	}

	// Print the image tensor
	// utils.ToPng("/tmp/object_detection.png", utils.TensorData(utils.TensorPtrC(tensor)), i.Bounds())

	// Transform the decoded YCbCr JPG image into RGBA
	b := i.Bounds()
	img := image.NewRGBA(b)
	draw.Draw(img, b, i, b.Min, draw.Src)

	// Input op
	inputop := graph.Operation("image_tensor")

	// Output ops
	o1 := graph.Operation("detection_boxes")
	o2 := graph.Operation("detection_scores")
	o3 := graph.Operation("detection_classes")
	o4 := graph.Operation("detection_masks")

	// Execute COCO Graph
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputop.Output(0): tensor,
		},
		[]tf.Output{
			o1.Output(0),
			o2.Output(0),
			o3.Output(0),
			o4.Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}

	// Take the first in the batched output
	boxes := output[0].Value().([][][]float32)[0]
	probabilities := output[1].Value().([][]float32)[0]
	classes := output[2].Value().([][]float32)[0]
	masks := output[3].Value().([][][][]float32)[0]

	// Draw a box around the objects with a probability higher than the threshold
	curObj := 0
	for probabilities[curObj] > 0.9 {
		y1 := float32(img.Bounds().Max.Y) * boxes[curObj][0]
		x1 := float32(img.Bounds().Max.X) * boxes[curObj][1]
		y2 := float32(img.Bounds().Max.Y) * boxes[curObj][2]
		x2 := float32(img.Bounds().Max.X) * boxes[curObj][3]

		mask := masks[curObj]
		color := colornames.Map[colornames.Names[int(classes[curObj])]]

		utils.Rect(img, int(x1), int(y1), int(x2), int(y2), 4, color)
		utils.AddLabel(img, int(x1), int(y1), int(classes[curObj]), utils.GetLabel(curObj, probabilities, classes, labels))
		img = utils.Segment(img, mask, color, x1, y1, x2, y2)
		curObj++
	}

	// Output JPG file
	outfile, err := os.Create(*outjpg)
	if err != nil {
		log.Fatal(err)
	}
	var opt jpeg.Options
	opt.Quality = 80
	err = jpeg.Encode(outfile, img, &opt)
	if err != nil {
		log.Fatal(err)
	}
}
