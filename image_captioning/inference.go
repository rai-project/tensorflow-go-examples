package main

import (
	"bytes"
	"container/heap"
	"fmt"
	"image"
	_ "image/jpeg"
	"io/ioutil"
	"log"
	"math"
	"path/filepath"
	"reflect"
	"sort"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func makeTensorFromImage(filename string) (*tf.Tensor, image.Image, int32, int32, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, nil, 0, 0, err
	}

	r := bytes.NewReader(b)
	img, _, err := image.Decode(r)
	if err != nil {
		fmt.Println("Input image decoding failed!!!")
		fmt.Println(err)
	}

	x := img.Bounds().Max.X
	y := img.Bounds().Max.Y

	if err != nil {
		return nil, nil, 0, 0, err
	}

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(b))
	if err != nil {
		return nil, nil, 0, 0, err
	}
	// Creates a tensorflow graph to decode the jpeg image
	graph, input, output, newX, newY, err := constructGraphToNormalizeImage(x, y)
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
		fmt.Println("input image conversion to tensor error", err)
		return nil, nil, 0, 0, err
	}
	return normalized[0], img, newX, newY, nil
}

func constructGraphToNormalizeImage(x, y int) (graph *tf.Graph, input, output tf.Output, H, W int32, err error) {
	inputSize := 346
	W = int32(inputSize)
	// H = int32(1.0 * y * inputSize / x)
	H = int32(346)
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output =
		op.ResizeBilinear(s,
			op.ExpandDims(s,
				op.Cast(s,
					op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
					tf.Float),
				op.Const(s.SubScope("make_batch"), int32(0))),
			op.Const(s.SubScope("size"), []int32{H, W}))
	graph, err = s.Finalize()
	return graph, input, output, W, H, err
}

// reference: https://github.com/galeone/tfgo/blob/master/ops.go
func batchify(scope *op.Scope, tensors []tf.Output) tf.Output {
	s := scope.SubScope("batchify")
	// Batchify a single value, means add batch dimension and return
	if len(tensors) == 1 {
		return op.ExpandDims(s.SubScope("ExpandDims"), tensors[0], op.Const(s.SubScope("axis"), []int32{0}))
	}
	var tensors4d []tf.Output
	for _, tensor := range tensors {
		tensors4d = append(tensors4d, op.ExpandDims(s.SubScope("ExpandDims"), tensor, op.Const(s.SubScope("axis"), []int32{0})))
	}
	return op.ConcatV2(s.SubScope("ConcatV2"), tensors4d, op.Const(s.SubScope("axis"), int32(0)))
}

type argSort struct {
	Args []float32
	Idxs []int
}

// Implement sort.Interface Len
func (s argSort) Len() int { return len(s.Args) }

// Implement sort.Interface Less
func (s argSort) Less(i, j int) bool { return s.Args[i] > s.Args[j] }

// Implment sort.Interface Swap
func (s argSort) Swap(i, j int) {
	// swap value
	s.Args[i], s.Args[j] = s.Args[j], s.Args[i]
	// swap index
	s.Idxs[i], s.Idxs[j] = s.Idxs[j], s.Idxs[i]
}

func main() {
	// Parse flags
	modeldir := "./"
	jpgfile := "./COCO_val2014_000000224477.jpg"
	// outjpg := "output.jpg"

	modelpath := filepath.Join(modeldir, "im2txt_frozen_graph.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}

	// Construct the vocabulary
	// vocab, reverseVocab := constructVocabulary("word_counts_p.txt")

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}
	fmt.Println(reflect.TypeOf(graph))

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		fmt.Println("create new session error", err)
		log.Fatal(err)
	}
	defer session.Close()

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, _, _, _, err := makeTensorFromImage(jpgfile)
	if err != nil {
		fmt.Println("makeTensorFromImage error", err)
		log.Fatal(err)
	}

	// Get all the input and output operations
	inputOp := graph.Operation("control_dependency")
	outputOp := graph.Operation("lstm/initial_state")

	// Execute COCO Graph
	initialState, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputOp.Output(0): tensor,
		},
		[]tf.Output{
			outputOp.Output(0),
		},
		nil)
	if err != nil {
		fmt.Println("session run error", err)
		log.Fatal(err)
	}

	// // Outputs
	// fmt.Println(reflect.TypeOf(output))
	// initialState := output[0].Value().([][]float32)
	// fmt.Println(reflect.TypeOf(initialState))
	// for i := 0; i < 256; i++ {
	// 	for j := 0; j < 4; j++ {
	// 		fmt.Printf("%f\t", initialState[0][i*4+j])
	// 	}
	// 	fmt.Printf("\n")
	// }

	// Beam Search
	beamSize := 3
	maxCaptionLength := 20
	lengthNormalizationFactor := 0.0

	vocabFilename := "./word_counts_p.txt"
	vocab := constructVocabulary(vocabFilename)

	initialBeam := caption{
		sentence: []int{vocab.startID},
		state:    *initialState[0],
		logprob:  0.0,
		score:    0.0}

	partialCaption := &topN{n: beamSize}
	heap.Init(partialCaption)
	completeCaption := &topN{n: beamSize}
	heap.Init(completeCaption)

	partialCaption.PushTopN(initialBeam)

	intermediateInputFeed := graph.Operation("input_feed")
	intermediateStateFeed := graph.Operation("lstm/state_feed")
	softmaxOp := graph.Operation("softmax")
	stateOp := graph.Operation("lstm/state")

	for i := 0; i < maxCaptionLength; i++ {
		partialCaptionList := partialCaption.Extract()
		partialCaption.Reset()

		inputFeed := []int{}
		stateFeed := []tf.Tensor{}
		for idx, cap := range partialCaptionList {
			inputFeed := []int{}
			inputFeed = append(inputFeed, cap.sentence[len(cap.sentence)-1])
			stateFeed = append(stateFeed, cap.state)
		}
		stateFeed := stackTensorList(stateFeed)

		softmaxOutput, stateOutput, err := session.Run(
			map[tf.Output]*tf.Tensor{
				intermediateInputFeed.Output(0): inputFeed,
				intermediateStateFeed.Output(0): stateFeed,
			},
			[]tf.Output{
				softmaxOp.Output(0),
				stateOp.Output(0),
			},
			nil)
		if err != nil {
			fmt.Println("intermediate session run error", err)
			log.Fatal(err)
		}

		for idx, cap := range partialCaptionList {
			wordProbabilities := softmaxOutput[idx].Value().([][]float32)[i]
			state := stateOutput[idx]

			wordLen := len(wordProbabilities)
			idxs := []int{}
			for i := 0; i < wordLen; i++ {
				idxs = append(idxs, i)
			}
			arg := argSort{Args: wordProbabilities, Idxs: idxs}
			sort.Sort(arg)
			mostLikelyWords := arg.Idxs[wordLen-beamSize : wordLen-1]

			for i := mostLikelyWords.Len(); i >= 0; i-- {
				w := mostLikelyWords[i]
				p := wordProbabilities[w]
				if p < 1e-12 {
					continue
				}

				sentence := append(cap.sentence, w)
				logprob := cap.logprob + float32(math.Log(p))
				score := logprob

				if w == vocab.endID {
					if lengthNormalizationFactor > 0 {
						score /= float32(math.Pow(float64(len(sentence)), lengthNormalizationFactor))
					}
					beam := caption{
						sentence: sentence,
						state:    state,
						logprob:  logprob,
						score:    score}
					completeCaption.PushTopN(beam)
				} else {
					beam := caption{
						sentence: sentence,
						state:    state,
						logprob:  logprob,
						score:    score}
					partialCaption.PushTopN(beam)
				}
			}
		}
		if partialCaption.Len() == 0 {
			break
		}
	}

	if completeCaption.Len() == 0 {
		completeCaption = partialCaption
	}

	captions := completeCaption.Extract()
	for i, caption := range captions {
		//print the final result
	}
}
