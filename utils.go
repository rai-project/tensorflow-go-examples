package utils

// #include <stdlib.h>
// #cgo LDFLAGS: -ltensorflow
// #cgo CFLAGS: -I${SRCDIR}/../../tensorflow/tensorflow
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"log"
	"os"
	"reflect"
	"unsafe"

	"github.com/disintegration/imaging"
	imagetypes "github.com/rai-project/image/types"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"golang.org/x/image/colornames"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// DRAWING UTILITY FUNCTIONS

// HLine draws a horizontal line
func HLine(img *image.RGBA, x1, y, x2 int, col color.Color) {
	for ; x1 <= x2; x1++ {
		img.Set(x1, y, col)
	}
}

// VLine draws a veritcal line
func VLine(img *image.RGBA, x, y1, y2 int, col color.Color) {
	for ; y1 <= y2; y1++ {
		img.Set(x, y1, col)
	}
}

// Rect draws a rectangle utilizing HLine() and VLine()
func Rect(img *image.RGBA, x1, y1, x2, y2, width int, col color.Color) {
	for i := 0; i < width; i++ {
		HLine(img, x1, y1+i, x2, col)
		HLine(img, x1, y2+i, x2, col)
		VLine(img, x1+i, y1, y2, col)
		VLine(img, x2+i, y1, y2, col)
	}
}

// Segment draws a rectangle utilizing HLine() and VLine()
func Segment(img *image.RGBA, mask [][]float32, col color.Color, x1, y1, x2, y2 float32) *image.RGBA {
	height := len(mask)
	width := len(mask[0])
	seg := image.NewRGBA(image.Rect(0, 0, width, height))

	for ii := 0; ii < height; ii++ {
		for jj := 0; jj < width; jj++ {
			if mask[ii][jj] > 0.2 {
				seg.Set(jj, ii, col)
			}
		}
	}

	segScaled := imaging.Resize(seg, int(x2)-int(x1), int(y2)-int(y1), imaging.NearestNeighbor)

	out, _ := os.Create("/tmp/test.png")
	defer out.Close()
	err := png.Encode(out, segScaled)
	if err != nil {
		log.Println(err)
	}

	overlay := imaging.Overlay(img, segScaled, image.Pt(int(x1), int(y1)), 0.5)
	rgba := &image.RGBA{
		Pix:    overlay.Pix,
		Stride: overlay.Stride,
		Rect:   overlay.Rect,
	}

	return rgba
}

func ToPng(filePath string, imgByte []byte, bounds image.Rectangle) {
	img := imagetypes.NewRGBImage(bounds)
	copy(img.Pix, imgByte)

	out, _ := os.Create(filePath)
	defer out.Close()

	err := png.Encode(out, img.ToRGBAImage())
	if err != nil {
		log.Println(err)
	}
}

// LABEL UTILITY FUNCTIONS

func LoadLabels(labelsFile string) []string {
	var labels []string
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}

	return labels
}

func GetLabel(idx int, probabilities []float32, classes []float32, labels []string) string {
	index := int(classes[idx])
	label := fmt.Sprintf("%s (%2.0f%%)", labels[index], probabilities[idx]*100.0)

	return label
}

func AddLabel(img *image.RGBA, x, y, class int, label string) {
	col := colornames.Map[colornames.Names[class]]
	point := fixed.Point26_6{fixed.Int26_6(x * 64), fixed.Int26_6(y * 64)}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(colornames.Black),
		Face: basicfont.Face7x13,
		Dot:  point,
	}

	Rect(img, x, y-13, (x + len(label)*7), y-6, 7, col)

	d.DrawString(label)
}

// TENSOR UTILITY FUNCTIONS

func DecodeJpegGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	dctMethod := op.DecodeJpegDctMethod("INTEGER_ACCURATE")
	// dctMethod := op.DecodeJpegDctMethod("INTEGER_FAST")
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3), dctMethod),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func DecodeJpegNormalizeGraph(height int32, width int32) (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	dctMethod := op.DecodeJpegDctMethod("INTEGER_ACCURATE")
	output =
		op.Cast(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.DecodeJpeg(s, input, op.DecodeJpegChannels(3), dctMethod),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{height, width})),
			tf.Uint8)
	graph, err = s.Finalize()
	return graph, input, output, err
}

func MakeTensorFromImage(filename string) (*tf.Tensor, image.Image, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, nil, err
	}

	r := bytes.NewReader(b)
	img, _, err := image.Decode(r)

	if err != nil {
		return nil, nil, err
	}

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(b))
	if err != nil {
		return nil, nil, err
	}
	// Creates a tensorflow graph to decode the jpeg image
	graph, input, output, err := DecodeJpegGraph()
	if err != nil {
		return nil, nil, err
	}
	// Execute that graph to decode this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, nil, err
	}
	return normalized[0], img, nil
}

func max(x, y int) int {
	if x < y {
		return y
	}
	return x
}

func MakeTensorFromResizedImage(filename string, inputSize int32) (*tf.Tensor, image.Image, int, int, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, nil, 0, 0, err
	}

	r := bytes.NewReader(b)
	img, _, err := image.Decode(r)
	width := img.Bounds().Max.X
	height := img.Bounds().Max.Y
	resizeRatio := float32(inputSize) / float32(max(width, height))
	targetWidth := int32(resizeRatio * float32(width))
	targetHeight := int32(resizeRatio * float32(height))

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(b))
	if err != nil {
		return nil, nil, 0, 0, err
	}

	// Creates a tensorflow graph to decode the jpeg image
	graph, input, output, err := DecodeJpegNormalizeGraph(targetHeight, targetWidth)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	// Execute that graph to decode this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	return normalized[0], img, int(targetWidth), int(targetHeight), nil
}

func ReshapeTensorFloats(data [][]float32, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]float32, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]float32, H)
		for h := int64(0); h < H; h++ {
			th := make([][]float32, W)
			for w := int64(0); w < W; w++ {
				offset := C * (W*h + w)
				tw := ndata[offset : offset+C]
				th[w] = tw
			}
			tn[h] = th
		}
		tensor[n] = tn
	}
	return tf.NewTensor(tensor)
}

func TensorPtrC(t *tf.Tensor) *C.TF_Tensor {
	fld := reflect.Indirect(reflect.ValueOf(t)).FieldByName("c")
	if fld.CanInterface() {
		return fld.Interface().(*C.TF_Tensor)
	}

	ptr := unsafe.Pointer(fld.UnsafeAddr())
	e := (**C.TF_Tensor)(ptr)
	return *e
}

func TensorData(c *C.TF_Tensor) []byte {
	// See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
	cbytes := C.TF_TensorData(c)
	if cbytes == nil {
		return nil
	}
	length := int(C.TF_TensorByteSize(c))
	slice := (*[1 << 30]byte)(unsafe.Pointer(cbytes))[:length:length]
	return slice
}

// IMAGE PREPROCESSING UTILITY FUNCTIONS

func NormalizeImageHWC(in *image.NRGBA, mean []float32, scale float32) ([]float32, error) {
	height := in.Bounds().Dy()
	width := in.Bounds().Dx()
	out := make([]float32, 3*height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			outOffset := 3 * (y*width + x)
			nrgba := in.NRGBAAt(x, y)
			r, g, b := nrgba.R, nrgba.G, nrgba.B
			out[outOffset+0] = (float32(r) - mean[0]) / scale
			out[outOffset+1] = (float32(g) - mean[1]) / scale
			out[outOffset+2] = (float32(b) - mean[2]) / scale
		}
	}
	return out, nil
}

// SORTING UTILITY FUNCTIONS

type Predictions struct {
	Indexes       []int
	Probabilities []float32
}

// Implement sort.Interface Len
func (s Predictions) Len() int { return len(s.Indexes) }

// Implement sort.Interface Less
func (s Predictions) Less(i, j int) bool { return s.Probabilities[i] > s.Probabilities[j] }

// Implment sort.Interface Swap
func (s Predictions) Swap(i, j int) {
	// swap value
	s.Probabilities[i], s.Probabilities[j] = s.Probabilities[j], s.Probabilities[i]
	// swap index
	s.Indexes[i], s.Indexes[j] = s.Indexes[j], s.Indexes[i]
}
