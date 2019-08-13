package main

import (
	"flag"
	"io/ioutil"
	"log"
	"path/filepath"
	"sort"

	"github.com/disintegration/imaging"
	"github.com/k0kubun/pp"
	utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	// Parse flags
	modeldir := flag.String("dir", "", "Directory containing trained model files. Assumes model file is called frozen_inference_graph.pb")
	jpgfile := flag.String("jpg", "platypus.jpg", "Path of a JPG image to use for input")
	labelfile := flag.String("labels", "synset1.txt", "Path to file of COCO labels, one per line")
	flag.Parse()
	if *modeldir == "" || *jpgfile == "" {
		flag.Usage()
		return
	}

	// Load the labels
	labels := utils.LoadLabels(*labelfile)

	// Load a frozen graph to use for queries
	modelpath := filepath.Join(*modeldir, "mobilenet_v1_1.0_224_frozen.pb")
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

	img, err := imaging.Open(*jpgfile)
	if err != nil {
		log.Fatalf("failed to open image: %v", err)
	}

	height := 224
	width := 224
	resized := imaging.Resize(img, width, height, imaging.Linear)
	imgFloats, err := utils.NormalizeImageHWC(resized, []float32{128, 128, 128}, 128)
	if err != nil {
		panic(err)
	}

	batchSize := 1
	input := make([][]float32, batchSize)
	for ii := 0; ii < batchSize; ii++ {
		input[ii] = imgFloats
	}

	tensor, err := utils.ReshapeTensorFloats(input, []int64{int64(batchSize), int64(height), int64(width), 3})
	if err != nil {
		log.Fatal(err)
	}

	// Input op
	inputOp := graph.Operation("input")

	// Output ops
	o1 := graph.Operation("MobilenetV1/Predictions/Reshape_1")

	// Execute COCO Graph
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputOp.Output(0): tensor,
		},
		[]tf.Output{
			o1.Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}
	// Take the first in the batched output
	probabilities := output[0].Value().([][]float32)[0]

	idxs := make([]int, len(probabilities))
	for i := range probabilities {
		idxs[i] = i
	}
	preds := utils.Predictions{Probabilities: probabilities, Indexes: idxs}
	sort.Sort(preds)

	for ii := 0; ii < 1; ii++ {
		pp.Println(preds.Indexes[ii], labels[preds.Indexes[ii]], preds.Probabilities[ii])
	}
}
