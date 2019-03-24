package main

import (
	"flag"
	"image"
	"image/color"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/disintegration/imaging"

	utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var LabelNames []string = []string{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
	"car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
	"sheep", "sofa", "train", "tv"}

func createPascalLabelColorMap() [256][3]int32 {
	var colorMap [256][3]int32
	var ind [256]int32
	for ii := 0; ii < 256; ii++ {
		ind[ii] = int32(ii)
	}
	for shift := 7; shift >= 0; shift-- {
		for jj := 0; jj < 256; jj++ {
			for kk := 0; kk < 3; kk++ {
				colorMap[jj][kk] |= ((ind[jj] >> uint(kk)) & 1) << uint(shift)
			}
		}
		for jj := range ind {
			ind[jj] >>= 3
		}
	}
	return colorMap
}

// // Crude colormap copied from python version. Only 21 entries because PASCAL VOC only has 21 classes
// var (
// 	colorMap = [21][3]uint8{{0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0},
// 		{0, 0, 128}, {128, 0, 128}, {0, 128, 128}, {128, 128, 128},
// 		{64, 0, 0}, {192, 0, 0}, {64, 128, 0}, {192, 128, 0},
// 		{64, 0, 128}, {192, 0, 128}, {64, 128, 128}, {192, 128, 128},
// 		{0, 64, 0}, {128, 64, 0}, {0, 192, 0}, {128, 192, 0}, {0, 64, 128}}
// )

func main() {
	// Parse flags
	modeldir := flag.String("dir", "", "Directory containing trained model files. Assumes model file is called frozen_inference_graph.pb")
	jpgfile := flag.String("jpg", "lane_control.jpg", "Path of a JPG image to use for input")
	outjpg := flag.String("out", "output.jpg", "Path of output JPG for displaying labels. Default is output.jpg")
	flag.Parse()
	if *modeldir == "" || *jpgfile == "" {
		flag.Usage()
		return
	}

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

	inputSize := 513
	tensor, img, targetWidth, targetHeight, err := utils.MakeTensorFromResizedImage(*jpgfile, int32(inputSize))
	if err != nil {
		log.Fatal(err)
	}
	// Input op
	inputOp := graph.Operation("ImageTensor")

	// Output ops
	outputOp := graph.Operation("SemanticPredictions")

	// Execute COCO Graph
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputOp.Output(0): tensor,
		},
		[]tf.Output{
			outputOp.Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}

	// Take the first in the batched output
	seg := output[0].Value().([][][]int64)[0]

	colorMap := createPascalLabelColorMap()
	// pp.Println(colorMap)
	imgSeg := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	for w := 0; w < targetWidth; w++ {
		for h := 0; h < targetHeight; h++ {
			v := seg[h][w]
			if v != 0 {
				R, G, B := uint8(colorMap[v][0]), uint8(colorMap[v][1]), uint8(colorMap[v][2])
				imgSeg.Set(w, h, color.RGBA{R, G, B, 255})
			}
		}
	}

	imgResized := imaging.Resize(img, targetWidth, targetHeight, imaging.Linear)
	imgOut := imaging.Overlay(imgResized, imgSeg, image.ZP, 0.5)

	// Output JPG file
	outfile, err := os.Create(*outjpg)
	if err != nil {
		log.Fatal(err)
	}
	var opt jpeg.Options
	opt.Quality = 80
	err = jpeg.Encode(outfile, imgOut, &opt)
	if err != nil {
		log.Fatal(err)
	}
}
